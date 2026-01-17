package main

import (
	"fmt"
	"index/suffixarray"
	"math"
	"math/rand"
	"os"
	"time"
)

// Sample samples the next character with temperature, trying progressively shorter suffixes.
// Returns the sampled byte and the n-gram length used (0 if no match found).
func Sample(idx *suffixarray.Index, context string, temp float64) (byte, int) {
	data := idx.Bytes()
	for i := 0; i < len(context); i++ {
		offsets := idx.Lookup([]byte(context[i:]), -1)
		if len(offsets) == 0 {
			continue
		}

		// Collect next characters
		var chars []byte
		contextLen := len(context) - i
		for _, offset := range offsets {
			nextPos := offset + contextLen
			if nextPos < len(data) {
				chars = append(chars, data[nextPos])
			}
		}
		if len(chars) == 0 {
			continue
		}

		// Count occurrences and apply temperature
		counts := make(map[byte]int)
		for _, ch := range chars {
			counts[ch]++
		}
		weights := make(map[byte]float64)
		var total float64
		for ch, cnt := range counts {
			w := math.Pow(float64(cnt), 1.0/temp)
			weights[ch] = w
			total += w
		}

		// Sample
		r := rand.Float64() * total
		for ch, w := range weights {
			r -= w
			if r < 0 {
				return ch, contextLen
			}
		}
	}
	return 0, 0
}

// SampleK samples from k different n-gram levels and combines their distributions.
// It finds k levels where each subsequent level has more matches than the previous.
// Weighting: longer matches (fewer counts) get higher weight via log(totalMatches)/log(levelMatches)
// This gives high specificity matches more influence while still mixing in broader context.
// Returns the sampled byte and the n-gram lengths used for each level.
func SampleK(idx *suffixarray.Index, context string, temp float64, k int) (byte, []int) {
	data := idx.Bytes()

	type level struct {
		counts     map[byte]int
		numMatches int
		n          int
	}
	var levels []level
	lastMatchCount := 0

	for i := 0; i < len(context) && len(levels) < k; i++ {
		offsets := idx.Lookup([]byte(context[i:]), -1)
		if len(offsets) == 0 {
			continue
		}

		// Collect next characters
		var chars []byte
		contextLen := len(context) - i
		for _, offset := range offsets {
			nextPos := offset + contextLen
			if nextPos < len(data) {
				chars = append(chars, data[nextPos])
			}
		}
		if len(chars) == 0 {
			continue
		}

		// Only add this level if it has more matches than the last
		if len(chars) > lastMatchCount {
			counts := make(map[byte]int)
			for _, ch := range chars {
				counts[ch]++
			}
			levels = append(levels, level{counts: counts, numMatches: len(chars), n: contextLen})
			lastMatchCount = len(chars)
		}
	}

	if len(levels) == 0 {
		return 0, nil
	}

	// Collect n values for each level
	nValues := make([]int, len(levels))
	for i, lvl := range levels {
		nValues[i] = lvl.n
	}

	// Combine distributions with count-based weighting
	// Weight = log(maxMatches) / log(levelMatches) so fewer matches = higher weight
	// This favors specific (long) matches while still mixing in general patterns
	combined := make(map[byte]float64)
	maxMatches := levels[len(levels)-1].numMatches

	for _, lvl := range levels {
		// Weight: more specific (fewer matches) gets higher weight
		// Using log ratio: log(max)/log(current) ranges from 1 (broadest) to higher values (more specific)
		var weight float64
		if lvl.numMatches == 1 {
			weight = math.Log(float64(maxMatches) + 1)
		} else {
			weight = math.Log(float64(maxMatches)+1) / math.Log(float64(lvl.numMatches)+1)
		}

		// Add weighted counts to combined distribution
		for ch, cnt := range lvl.counts {
			combined[ch] += weight * float64(cnt)
		}
	}

	// Apply temperature and normalize
	weights := make(map[byte]float64)
	var total float64
	for ch, w := range combined {
		adjusted := math.Pow(w, 1.0/temp)
		weights[ch] = adjusted
		total += adjusted
	}

	// Sample
	r := rand.Float64() * total
	for ch, w := range weights {
		r -= w
		if r < 0 {
			return ch, nValues
		}
	}
	return 0, nil
}

// NStats holds statistics about n-gram lengths used
type NStats struct {
	Values []int
	Mean   float64
	Var    float64
}

func calcStats(values []int) NStats {
	if len(values) == 0 {
		return NStats{}
	}
	sum := 0
	for _, v := range values {
		sum += v
	}
	mean := float64(sum) / float64(len(values))

	varSum := 0.0
	for _, v := range values {
		diff := float64(v) - mean
		varSum += diff * diff
	}
	variance := varSum / float64(len(values))

	return NStats{Values: values, Mean: mean, Var: variance}
}

// Generate generates text until maxChars is reached.
// Returns the generated text and statistics about n-gram lengths used.
func Generate(idx *suffixarray.Index, prompt string, maxChars int, temp float64) (string, NStats) {
	result := []byte(prompt)
	var nValues []int
	for len(result) < maxChars {
		contextStart := 0
		if len(result) > 200 {
			contextStart = len(result) - 200
		}
		ch, n := Sample(idx, string(result[contextStart:]), temp)
		if ch == 0 {
			break
		}
		result = append(result, ch)
		nValues = append(nValues, n)
	}
	return string(result), calcStats(nValues)
}

// KStats holds statistics about n-gram lengths for each k level
type KStats struct {
	PerLevel []NStats // stats for each k level (index 0 = longest match, etc.)
}

// GenerateK generates text using k-level sampling until maxChars is reached.
// Returns the generated text and statistics about n-gram lengths for each k level.
func GenerateK(idx *suffixarray.Index, prompt string, maxChars int, temp float64, k int) (string, KStats) {
	result := []byte(prompt)
	// Track n values per level: levelNs[i] contains all n values seen for level i
	levelNs := make([][]int, k)
	for i := range levelNs {
		levelNs[i] = []int{}
	}

	for len(result) < maxChars {
		contextStart := 0
		if len(result) > 200 {
			contextStart = len(result) - 200
		}
		ch, nValues := SampleK(idx, string(result[contextStart:]), temp, k)
		if ch == 0 {
			break
		}
		result = append(result, ch)
		// Record n values for each level that was used
		for i, n := range nValues {
			if i < k {
				levelNs[i] = append(levelNs[i], n)
			}
		}
	}

	// Calculate stats for each level
	stats := KStats{PerLevel: make([]NStats, k)}
	for i := 0; i < k; i++ {
		stats.PerLevel[i] = calcStats(levelNs[i])
	}
	return string(result), stats
}

func main() {
	data, err := os.ReadFile("data.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	idx := suffixarray.New(data)

	k := 3

	// Compare original vs k-level sampling
	fmt.Println("=== Original (longest match only) ===")
	start := time.Now()
	output, stats := Generate(idx, "First Citizen:", 1000, 0.8)
	fmt.Println(output)
	fmt.Printf("\nGenerated %d chars in %.4fs\n", len(output), time.Since(start).Seconds())
	fmt.Printf("N-gram stats: mean=%.2f, variance=%.2f\n", stats.Mean, stats.Var)

	fmt.Printf("\n=== K-level sampling (k=%d) ===\n", k)
	start = time.Now()
	outputK, kstats := GenerateK(idx, "First Citizen:", 1000, 0.8, k)
	fmt.Println(outputK)
	fmt.Printf("\nGenerated %d chars in %.4fs\n", len(outputK), time.Since(start).Seconds())
	fmt.Println("N-gram stats per level:")
	for i, s := range kstats.PerLevel {
		if len(s.Values) > 0 {
			fmt.Printf("  Level %d (k=%d): mean=%.2f, variance=%.2f, samples=%d\n", i+1, i+1, s.Mean, s.Var, len(s.Values))
		}
	}
}
