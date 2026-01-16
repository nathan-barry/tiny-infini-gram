package main

import (
	"fmt"
	"index/suffixarray"
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

// Sample samples a token from the distribution, trying progressively shorter suffixes
func Sample(idx *suffixarray.Index, context string) string {
	for i := 0; i < len(context); i++ {
		suffix := context[i:]
		counts := GetNextTokenCounts(idx, suffix)
		if len(counts) == 0 {
			continue
		}

		// Calculate total and sample
		var total int
		for _, cnt := range counts {
			total += cnt
		}
		r := rand.Intn(total)
		for tok, cnt := range counts {
			r -= cnt
			if r < 0 {
				return tok
			}
		}
	}
	return ""
}

// Generate generates text by repeatedly sampling the next token
func Generate(idx *suffixarray.Index, prompt string, maxTokens int) string {
	result := prompt
	for i := 0; i < maxTokens; i++ {
		contextStart := 0
		if len(result) > 200 {
			contextStart = len(result) - 200
		}
		token := Sample(idx, result[contextStart:])
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
	fmt.Println(Generate(idx, "MARCIUS:", 1000))
}
