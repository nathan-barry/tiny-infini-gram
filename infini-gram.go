package main

import (
	"fmt"
	"index/suffixarray"
	"math"
	"math/rand"
	"os"
	"sort"
	"time"
)

// Level represents an n-gram match level.
type Level struct {
	counts     map[byte]int // next char -> frequency
	numMatches int          // total matches at this level
	n          int          // context length (n-gram size)
}

// Sample returns the next byte sampled from k n-gram levels, plus the n and numMatches at each level.
// k=-1 uses all levels (down to n=1).
func Sample(idx *suffixarray.Index, context string, temp float64, k int) (byte, []int, []int) {
	data := idx.Bytes()
	var levels []Level
	lastNumMatches := 0

	for i := 0; i < len(context) && (k < 0 || len(levels) < k); i++ {
		offsets := idx.Lookup([]byte(context[i:]), -1)
		if len(offsets) == 0 {
			continue
		}
		counts := make(map[byte]int)
		n := len(context) - i
		for _, off := range offsets {
			if pos := off + n; pos < len(data) {
				counts[data[pos]]++
			}
		}
		numMatches := 0
		for _, c := range counts {
			numMatches += c
		}
		if numMatches > lastNumMatches {
			levels = append(levels, Level{counts, numMatches, n})
			lastNumMatches = numMatches
		}
	}
	if len(levels) == 0 {
		return 0, nil, nil
	}

	// Combine distributions: exponential decay by level index
	combined := make(map[byte]float64)
	nValues := make([]int, len(levels))
	matchCounts := make([]int, len(levels))
	decay := 0.1
	for i, lvl := range levels {
		nValues[i] = lvl.n
		matchCounts[i] = lvl.numMatches
		w := math.Pow(decay, float64(i))
		for ch, cnt := range lvl.counts {
			combined[ch] += w * float64(cnt)
		}
	}

	// Apply temperature and sample
	var total float64
	for ch, w := range combined {
		combined[ch] = math.Pow(w, 1/temp)
		total += combined[ch]
	}
	r := rand.Float64() * total
	for ch, w := range combined {
		if r -= w; r < 0 {
			return ch, nValues, matchCounts
		}
	}
	return 0, nil, nil
}

// LevelStats holds mean, std, and median for n and numMatches at a level.
type LevelStats struct {
	NMean, NStd, NMedian           float64
	MatchMean, MatchStd, MatchMedian float64
}

// Generate produces text and returns stats for n and numMatches at each level.
func Generate(idx *suffixarray.Index, prompt string, maxChars int, temp float64, k int) (string, []LevelStats) {
	result := []byte(prompt)
	var levelNs [][]int
	var levelMatches [][]int

	for len(result) < maxChars {
		start := max(0, len(result)-200)
		ch, ns, matches := Sample(idx, string(result[start:]), temp, k)
		if ch == 0 {
			break
		}
		result = append(result, ch)
		for i, n := range ns {
			for len(levelNs) <= i {
				levelNs = append(levelNs, nil)
			}
			levelNs[i] = append(levelNs[i], n)
		}
		for i, m := range matches {
			for len(levelMatches) <= i {
				levelMatches = append(levelMatches, nil)
			}
			levelMatches[i] = append(levelMatches[i], m)
		}
	}

	stats := make([]LevelStats, max(len(levelNs), len(levelMatches)))
	for i := range stats {
		if i < len(levelNs) && len(levelNs[i]) > 0 {
			stats[i].NMean, stats[i].NStd, stats[i].NMedian = meanStdMedian(levelNs[i])
		}
		if i < len(levelMatches) && len(levelMatches[i]) > 0 {
			stats[i].MatchMean, stats[i].MatchStd, stats[i].MatchMedian = meanStdMedian(levelMatches[i])
		}
	}
	return string(result), stats
}

func meanStdMedian(vals []int) (float64, float64, float64) {
	if len(vals) == 0 {
		return 0, 0, 0
	}
	var sum int
	for _, v := range vals {
		sum += v
	}
	mean := float64(sum) / float64(len(vals))
	var varSum float64
	for _, v := range vals {
		varSum += (float64(v) - mean) * (float64(v) - mean)
	}
	sorted := make([]int, len(vals))
	copy(sorted, vals)
	sort.Ints(sorted)
	var median float64
	if len(sorted)%2 == 0 {
		median = float64(sorted[len(sorted)/2-1]+sorted[len(sorted)/2]) / 2
	} else {
		median = float64(sorted[len(sorted)/2])
	}
	return mean, math.Sqrt(varSum / float64(len(vals))), median
}

func main() {
	data, _ := os.ReadFile("data.txt")
	idx := suffixarray.New(data)
	k := 2

	start := time.Now()
	output, stats := Generate(idx, "First Citizen:", 1000, 0.8, k)
	fmt.Println(output)
	fmt.Printf("\nGenerated %d chars in %.4fs\n", len(output), time.Since(start).Seconds())
	for i, s := range stats {
		if s.NMean > 0 {
			fmt.Printf("  Level %d: n(med=%.1f, avg=%.2f, std=%.2f) m(med=%.1f, avg=%.1f, std=%.1f)\n",
				i+1, s.NMedian, s.NMean, s.NStd, s.MatchMedian, s.MatchMean, s.MatchStd)
		}
	}
}
