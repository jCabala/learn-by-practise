package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"../lib"
)

func main() {
	src := flag.String("src", "", "The source URL to start the sitemap.")
	dest := flag.String("out", "sitemap.xml", "The sitemap file to be created. Must be in xml format.")
	depth := flag.Int("depth", 5, "The depth of the sitemap. Must be positive.")

	flag.Parse()

	validateSrc(src)
	validateDest(dest)
	validateDepth(depth)

	sm, err := sitemap.NewSitemap(*src, *depth)
	handleError(err)

	out, err := os.Create(*dest)
	handleError(err)
	
	defer out.Close()
	sm.WriteXml(out)
	fmt.Println("Sitemap created at", *dest)
}

func handleError(err error) {
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func validateSrc(src *string) {
	if *src == "" {
		flag.Usage()
		os.Exit(1)
	}
}

func validateDest(dest *string) {
	if *dest == "" ||  !strings.HasSuffix(strings.ToLower(*dest), ".xml"){
		flag.Usage()
		os.Exit(1)
	}
}

func validateDepth(depth *int) {
	if *depth <= 0 {
		flag.Usage()
		os.Exit(1)
	}
}