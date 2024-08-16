package main

import (
	"os"
	"../lib"
	"fmt"
)

func main() {
	fmt.Println("Starting the program...")
	for i := 1; i <= 4; i++ {
		fileName := fmt.Sprintf("ex%d.html", i)
		fmt.Println("Searching file" + fileName + "...")
		r, _ := os.Open("../test_examples/" + fileName)
		links, err := link.Parse(r)

		if err != nil {
			fmt.Println(err)
			return
		}
		
		fmt.Println("Here are the links: ")
		for j, link := range links {
			fmt.Printf("%d: Href: %s, Text: %s\n", j, link.Href, link.Text)
		}
		fmt.Println()
		fmt.Println()
	}
}