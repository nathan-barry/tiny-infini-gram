package main

import (
	"fmt"
	"index/suffixarray"
	"math"
	"math/rand"
	"os"
	"time"
)

// Sample samples the next character with temperature, trying progressively shorter suffixes
func Sample(idx *suffixarray.Index, context string, temp float64) byte {
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
				return ch
			}
		}
	}
	return 0
}

// Generate generates text until maxChars is reached
func Generate(idx *suffixarray.Index, prompt string, maxChars int, temp float64) string {
	result := []byte(prompt)
	for len(result) < maxChars {
		contextStart := 0
		if len(result) > 200 {
			contextStart = len(result) - 200
		}
		ch := Sample(idx, string(result[contextStart:]), temp)
		if ch == 0 {
			break
		}
		result = append(result, ch)
	}
	return string(result)
}

func main() {
	data, err := os.ReadFile("data.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	idx := suffixarray.New(data)
	start := time.Now()
	output := Generate(idx, "First Citizen:", 2000, 0.8)
	fmt.Println(output)
	fmt.Printf("\nGenerated %d chars in %.4fs\n", len(output), time.Since(start).Seconds())
}
