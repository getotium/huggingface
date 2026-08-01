package huggingface

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestSearch(t *testing.T) {
	var gotPath, gotQuery, gotUA string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotUA = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "application/json")
		// Two rows exercising the polymorphic gated field: bool false and string "manual".
		_, _ = w.Write([]byte(`[
			{"id":"Qwen/Qwen3-8B","author":"Qwen","downloads":12,"likes":3,"gated":false,
			 "pipeline_tag":"text-generation","library_name":"transformers","createdAt":"2025-04-27T03:42:21.000Z",
			 "tags":["conversational","license:apache-2.0"]},
			{"id":"meta-llama/Llama-3.1-8B-Instruct","author":"meta-llama","downloads":99,"likes":50,"gated":"manual",
			 "pipeline_tag":"text-generation","tags":["license:llama3.1"]}
		]`))
	}))
	defer srv.Close()

	c := New(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()), WithUserAgent("otium-test/1.0"))
	models, err := c.Search(context.Background(), SearchOptions{
		Search: "qwen", Author: "Qwen", PipelineTag: "text-generation", Sort: "downloads", Direction: -1, Limit: 5, Full: true,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if gotPath != "/api/models" {
		t.Errorf("path = %q, want /api/models", gotPath)
	}
	for _, want := range []string{"search=qwen", "author=Qwen", "pipeline_tag=text-generation", "sort=downloads", "direction=-1", "limit=5", "full=true"} {
		if !strings.Contains(gotQuery, want) {
			t.Errorf("query %q missing %q", gotQuery, want)
		}
	}
	if gotUA != "otium-test/1.0" {
		t.Errorf("User-Agent = %q, want otium-test/1.0", gotUA)
	}
	if len(models) != 2 {
		t.Fatalf("got %d models, want 2", len(models))
	}
	if models[0].Gated.IsGated() {
		t.Errorf("Qwen3-8B should be open (gated=false)")
	}
	if !models[1].Gated.IsGated() || models[1].Gated != "manual" {
		t.Errorf("Llama gated = %q, want manual", models[1].Gated)
	}
}

func TestModel(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/Qwen/Qwen3-8B" {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"Qwen/Qwen3-8B","gated":false,
			"safetensors":{"total":8190735360,"parameters":{"BF16":8190735360}}}`))
	}))
	defer srv.Close()

	c := New(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	m, err := c.Model(context.Background(), "Qwen/Qwen3-8B")
	if err != nil {
		t.Fatalf("Model: %v", err)
	}
	if m.Safetensors == nil {
		t.Fatal("safetensors is nil")
	}
	if m.Safetensors.Total != 8190735360 {
		t.Errorf("total = %d, want 8190735360", m.Safetensors.Total)
	}
	if dt := m.Safetensors.DominantDtype(); dt != "BF16" {
		t.Errorf("dominant dtype = %q, want BF16", dt)
	}
}

func TestModelNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "not found", http.StatusNotFound)
	}))
	defer srv.Close()

	c := New(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if _, err := c.Model(context.Background(), "nope/nope"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
}

func TestRateLimited(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "slow down", http.StatusTooManyRequests)
	}))
	defer srv.Close()

	c := New(WithBaseURL(srv.URL), WithHTTPClient(srv.Client()))
	if _, err := c.Search(context.Background(), SearchOptions{}); !errors.Is(err, ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
}
