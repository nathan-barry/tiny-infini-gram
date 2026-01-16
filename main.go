package main

import (
	"fmt"
	"index/suffixarray"
	"math"
	"math/rand"
	"os"
)

// GetNextTokenCounts returns possible next tokens (with leading whitespace) and their counts
func GetNextTokenCounts(idx *suffixarray.Index, context string) map[string]int {
	counts := make(map[string]int)
	offsets := idx.Lookup([]byte(context), -1)
	if len(offsets) == 0 {
		return counts
	}

	data := idx.Bytes()
	contextLen := len(context)
	for _, offset := range offsets {
		nextPos := offset + contextLen
		if nextPos >= len(data) {
			continue
		}

		// Include whitespace as part of the token
		start := nextPos
		end := start
		// Consume whitespace
		for end < len(data) && (data[end] == ' ' || data[end] == '\t' || data[end] == '\n' || data[end] == '\r') {
			end++
		}
		// Consume word
		for end < len(data) && data[end] != ' ' && data[end] != '\t' && data[end] != '\n' && data[end] != '\r' && end-start < 50 {
			end++
		}
		if end > start {
			counts[string(data[start:end])]++
		}
	}
	return counts
}

// Sample samples a token from the distribution with temperature
func Sample(idx *suffixarray.Index, context string, temp float64) string {
	for i := 0; i < len(context); i++ {
		suffix := context[i:]
		counts := GetNextTokenCounts(idx, suffix)
		if len(counts) == 0 {
			continue
		}

		// Apply temperature scaling
		weights := make(map[string]float64)
		var total float64
		for tok, cnt := range counts {
			w := math.Pow(float64(cnt), 1.0/temp)
			weights[tok] = w
			total += w
		}

		// Sample from scaled distribution
		r := rand.Float64() * total
		for tok, w := range weights {
			r -= w
			if r < 0 {
				return tok
			}
		}
	}
	return ""
}

// Generate generates text by repeatedly sampling the next token
func Generate(idx *suffixarray.Index, prompt string, maxTokens int, temp float64) string {
	result := prompt
	for i := 0; i < maxTokens; i++ {
		contextStart := 0
		if len(result) > 200 {
			contextStart = len(result) - 200
		}
		token := Sample(idx, result[contextStart:], temp)
		if token == "" {
			break
		}
		result += token
	}
	return result
}

func main() {
	data, err := os.ReadFile("data.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	idx := suffixarray.New(data)
	fmt.Println(Generate(idx, "MARCIUS:", 1000, 0.8))
}
