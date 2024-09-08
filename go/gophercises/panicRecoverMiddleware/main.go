package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
)

type panicWriter struct {
	hw http.ResponseWriter
	rw io.ReadWriter
}

func (pw *panicWriter) Header() http.Header {
	return pw.hw.Header()
}

func (pw *panicWriter) Write(b []byte) (int, error) {
	return pw.rw.Write(b)
}

func (pw *panicWriter) WriteHeader(statusCode int) {
	pw.hw.WriteHeader(statusCode)
}

func (pw *panicWriter) WriteHttp() {
	fmt.Fprint(pw.hw, pw.rw)
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/panic/", panicRecoveryMiddleware(panicDemo))
	mux.HandleFunc("/panic-after/", panicRecoveryMiddleware(panicAfterDemo))
	mux.HandleFunc("/", panicRecoveryMiddleware(hello))
	log.Fatal(http.ListenAndServe(":3000", mux))
}

func panicRecoveryMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contentWriter := panicWriter{hw: w}
		defer func() {
			if err := recover(); err != nil {
				http.Error(w, "Something went wrong", http.StatusInternalServerError)
			} else {
				
			}
		}()
		next.ServeHTTP(contentWriter, r)
	})
}

func panicDemo(w http.ResponseWriter, r *http.Request) {
	funcThatPanics()
}

func panicAfterDemo(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "<h1>Hello!</h1>")
	funcThatPanics()
}

func funcThatPanics() {
	panic("Oh no!")
}

func hello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "<h1>Hello!</h1>")
}
