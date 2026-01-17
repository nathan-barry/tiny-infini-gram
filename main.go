package main

import (
	"fmt"
	"index/suffixarray"
	"math"
	"math/rand"
	"os"
	"time"
)

// Level represents an n-gram match level.
type Level struct {
	counts     map[byte]int // next char -> frequency
	numMatches int          // total matches at this level
	n          int          // context length (n-gram size)
}

// Sample returns the next byte sampled from k n-gram levels, plus the n used at each level.
func Sample(idx *suffixarray.Index, context string, temp float64, k int) (byte, []int) {
	data := idx.Bytes()
	var levels []Level
	lastNumMatches := 0

	for i := 0; i < len(context) && len(levels) < k; i++ {
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
		return 0, nil
	}

	// Combine distributions: weight by log(maxNumMatches)/log(numMatches)
	combined := make(map[byte]float64)
	maxNumMatches := float64(levels[len(levels)-1].numMatches)
	nValues := make([]int, len(levels))
	for i, lvl := range levels {
		nValues[i] = lvl.n
		w := math.Log(maxNumMatches+1) / math.Max(1, math.Log(float64(lvl.numMatches)+1))
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
			return ch, nValues
		}
	}
	return 0, nil
}

// Generate produces text and returns stats (mean, std) for n at each level.
func Generate(idx *suffixarray.Index, prompt string, maxChars int, temp float64, k int) (string, []struct{ Mean, Std float64 }) {
	result := []byte(prompt)
	levelNs := make([][]int, k)

	for len(result) < maxChars {
		start := max(0, len(result)-200)
		ch, ns := Sample(idx, string(result[start:]), temp, k)
		if ch == 0 {
			break
		}
		result = append(result, ch)
		for i, n := range ns {
			levelNs[i] = append(levelNs[i], n)
		}
	}

	stats := make([]struct{ Mean, Std float64 }, k)
	for i, vals := range levelNs {
		if len(vals) == 0 {
			continue
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
		stats[i] = struct{ Mean, Std float64 }{mean, math.Sqrt(varSum / float64(len(vals)))}
	}
	return string(result), stats
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
		if s.Mean > 0 {
			fmt.Printf("  Level %d: mean=%.2f, std=%.2f\n", i+1, s.Mean, s.Std)
		}
	}
}
