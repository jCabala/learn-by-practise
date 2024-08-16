package link

import (
	"io"
	"golang.org/x/net/html"
)

type Link struct{
	Href string
	Text string
}

func Parse(r io.Reader) ([]Link, error) {
	t, err := html.Parse(r)

	if err != nil {	return nil, err }
	linkNodes := getLinkNodes(t)

	links := make([]Link, len(linkNodes))
	for i, node := range linkNodes {
		var link = Link{}
		link.Href = getHref(node)
		link.Text = getText(node)
		links[i] = link
	}

	return links, nil
}

func getHref(node *html.Node) string {
	for _, attr := range node.Attr {
		if attr.Key == "href" {
			return attr.Val
		}
	}

	return ""
}

func getText(n *html.Node) string {
	if n.Type == html.TextNode {
		return n.Data
	}

	if n.Type != html.ElementNode {
		return ""
	}

	var text string
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		text += getText(c)
	}

	return text
}

func getLinkNodes(n *html.Node) []*html.Node {
	if n.Type == html.ElementNode && n.Data == "a" {
		return []*html.Node{n}
	}

	var linkNodes []*html.Node
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		linkNodes = append(linkNodes, getLinkNodes(c)...)
	}

	return linkNodes
}
