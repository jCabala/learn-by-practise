package sitemap

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"net/url"
	"strings"

	"../../parser/lib"
)

type Sitemap struct {
	Pages []string
}

func NewSitemap(src string, depth int) (*Sitemap, error) {
	visited := map[string]bool{}

	baseUrl, err := getBaseUrl(src)
	if err != nil { return nil, err }

	crawl(src, depth, baseUrl, &visited)

	pages := []string{}
	for k := range visited {
		pages = append(pages, k)
	}

	return &Sitemap{pages}, nil
}

func crawl(url string, depth int, baseUrl string, visited *map[string]bool) {
		if depth == 0 || (*visited)[url] {
			return
		}

		(*visited)[url] = true
		links := getInternalLinks(url, baseUrl)

		for _, link := range links {
			crawl(link, depth-1, baseUrl, visited)
		}
	}


func getInternalLinks(str, baseUrl string) []string {
	res, err := http.Get(str)
	if err != nil {
		return []string{}
	}

	body, err := io.ReadAll(res.Body)
	res.Body.Close()
	if res.StatusCode > 299 {
		return []string{}	
	}
	if err != nil {
		return []string{}
	}
	
	links, err := link.Parse(bytes.NewReader(body))

	if err != nil {
		return []string{}
	}

	internalLinks := []string{}
	for _, link := range links {
		if strings.HasPrefix(link.Href, baseUrl) {
			internalLinks = append(internalLinks, link.Href)
		} else if strings.HasPrefix(link.Href, "/") {
			internalLinks = append(internalLinks, baseUrl + link.Href)
		}
	}

	return internalLinks
}

func getBaseUrl(str string) (string, error) {
	if (!isValidURL(str)) {
		return "", errors.New("invalid URL")
	}	

	parsedURL, _ := url.ParseRequestURI(str)

	host := strings.TrimPrefix(parsedURL.Host, "www.")

	baseURL := parsedURL.Scheme + "://" + host + parsedURL.Path

	return baseURL, nil
}

func isValidURL(str string) bool {
	parsedURL, err := url.ParseRequestURI(str)
	if err != nil {
		return false
	}

	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" { 
		return false
	}

	return parsedURL.Host != ""
}

func (s *Sitemap) WriteXml(out io.Writer) {
	out.Write([]byte("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"))
	out.Write([]byte("<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">\n"))
	for _, page := range s.Pages {
		out.Write([]byte("\t<url>\n"))
		out.Write([]byte("\t\t<loc>" + page + "</loc>\n"))
		out.Write([]byte("\t</url>\n"))
	}
	out.Write([]byte("</urlset>"))
}