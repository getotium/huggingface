// Package huggingface is a small, dependency-free client for the Hugging Face Hub
// API (https://huggingface.co/api). It exposes model search and per-model lookup and
// returns Hub-native types — it knows nothing about any particular application's
// domain (no model catalogs, VRAM math, or instance types here).
//
// It is intentionally self-contained (standard library only) so it can be lifted into
// a shared toolkit or its own repository unchanged. Construct a Client with New and
// functional options; the base URL, HTTP client, and User-Agent are all injectable so
// callers can point it at a test server and identify themselves as good API citizens.
package huggingface

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// DefaultBaseURL is the public Hugging Face Hub.
const DefaultBaseURL = "https://huggingface.co"

// defaultUserAgent identifies the library when a caller doesn't set its own. Callers
// are encouraged to override it with WithUserAgent so the Hub can attribute traffic.
const defaultUserAgent = "go-huggingface/0.1 (+https://github.com/getotium/huggingface)"

// Sentinel errors callers can match with errors.Is.
var (
	// ErrNotFound is returned when a model id does not exist (HTTP 404).
	ErrNotFound = errors.New("huggingface: not found")
	// ErrRateLimited is returned when the Hub asks us to slow down (HTTP 429).
	ErrRateLimited = errors.New("huggingface: rate limited")
)

// Client talks to the Hugging Face Hub API. It is safe for concurrent use.
type Client struct {
	http      *http.Client
	baseURL   string
	userAgent string
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient sets the underlying HTTP client (e.g. to inject a timeout, transport,
// or a test server's client).
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) {
		if h != nil {
			c.http = h
		}
	}
}

// WithBaseURL overrides the API base URL (default DefaultBaseURL). Trailing slashes are
// trimmed.
func WithBaseURL(u string) Option {
	return func(c *Client) {
		if u != "" {
			c.baseURL = strings.TrimRight(u, "/")
		}
	}
}

// WithUserAgent sets the User-Agent header. Identifying your application is good Hub
// etiquette and helps the maintainers reach you if your traffic misbehaves.
func WithUserAgent(ua string) Option {
	return func(c *Client) {
		if ua != "" {
			c.userAgent = ua
		}
	}
}

// New constructs a Client. With no options it targets the public Hub with a 15s
// timeout and the library's default User-Agent.
func New(opts ...Option) *Client {
	c := &Client{
		http:      &http.Client{Timeout: 15 * time.Second},
		baseURL:   DefaultBaseURL,
		userAgent: defaultUserAgent,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Safetensors describes a model's parameter counts as reported by the Hub's
// safetensors metadata. Total is the overall parameter count; Parameters breaks it down
// by tensor dtype (e.g. {"BF16": 8190735360}).
type Safetensors struct {
	Total      int64            `json:"total"`
	Parameters map[string]int64 `json:"parameters"`
}

// DominantDtype returns the dtype holding the most parameters (e.g. "BF16"), or "" if
// unknown. It is the dtype a caller would assume for a size estimate.
func (s *Safetensors) DominantDtype() string {
	if s == nil {
		return ""
	}
	var (
		best  string
		bestN int64 = -1
	)
	for dtype, n := range s.Parameters {
		if n > bestN {
			best, bestN = dtype, n
		}
	}
	return best
}

// Gated captures the Hub's polymorphic "gated" field, which is either the JSON boolean
// false (open) or a string mode ("auto"/"manual") for gated repos. It unmarshals both
// into a string: "" means open, otherwise the mode.
type Gated string

// UnmarshalJSON accepts either a bool or a string.
func (g *Gated) UnmarshalJSON(b []byte) error {
	var asBool bool
	if err := json.Unmarshal(b, &asBool); err == nil {
		if asBool {
			*g = "gated"
		} else {
			*g = ""
		}
		return nil
	}
	var asString string
	if err := json.Unmarshal(b, &asString); err == nil {
		*g = Gated(asString)
		return nil
	}
	return fmt.Errorf("huggingface: cannot parse gated field %q", string(b))
}

// IsGated reports whether the repo requires access approval.
func (g Gated) IsGated() bool { return g != "" }

// ModelInfo is a model record as returned by the Hub. Fields not requested or not
// applicable to a given endpoint are zero (notably Safetensors is nil on the search
// list endpoint; fetch a single model to populate it).
type ModelInfo struct {
	ID           string       `json:"id"`
	SHA          string       `json:"sha"` // the repo's current commit hash — pin this for reproducible pulls
	Author       string       `json:"author"`
	PipelineTag  string       `json:"pipeline_tag"`
	LibraryName  string       `json:"library_name"`
	Gated        Gated        `json:"gated"`
	Downloads    int          `json:"downloads"`
	Likes        int          `json:"likes"`
	CreatedAt    time.Time    `json:"createdAt"`
	LastModified time.Time    `json:"lastModified"`
	Tags         []string     `json:"tags"`
	Safetensors  *Safetensors `json:"safetensors"`
}

// SearchOptions parameterizes a model search. Zero-value fields are omitted from the
// request, yielding the Hub's defaults.
type SearchOptions struct {
	Search      string   // free-text query
	Author      string   // restrict to an author/org
	PipelineTag string   // e.g. "text-generation"
	Filter      string   // tag filter, e.g. "gguf"
	Sort        string   // e.g. "downloads", "likes", "trendingScore", "createdAt"
	Direction   int      // -1 descending, 1 ascending; 0 omits
	Limit       int      // max results; 0 omits
	Full        bool     // request full metadata
	Expand      []string // request specific fields via expand[]=; e.g. "safetensors", "tags".
	// Note: the Hub treats Expand as mutually exclusive with Full — when expand[] is
	// present, full is ignored and only the listed fields (plus id) are returned.
	Cursor string // pagination token from a prior page's Link header; "" for the first page
}

func (o SearchOptions) values() url.Values {
	v := url.Values{}
	set := func(key, val string) {
		if val != "" {
			v.Set(key, val)
		}
	}
	set("search", o.Search)
	set("author", o.Author)
	set("pipeline_tag", o.PipelineTag)
	set("filter", o.Filter)
	set("sort", o.Sort)
	if o.Direction != 0 {
		v.Set("direction", strconv.Itoa(o.Direction))
	}
	if o.Limit > 0 {
		v.Set("limit", strconv.Itoa(o.Limit))
	}
	if o.Full {
		v.Set("full", "true")
	}
	for _, e := range o.Expand {
		v.Add("expand[]", e)
	}
	set("cursor", o.Cursor)
	return v
}

// Search returns models matching opts (GET /api/models). It is the cursor-less
// convenience wrapper over SearchPage; use SearchPage to paginate.
func (c *Client) Search(ctx context.Context, opts SearchOptions) ([]ModelInfo, error) {
	models, _, err := c.SearchPage(ctx, opts)
	return models, err
}

// SearchPage returns one page of models matching opts along with the cursor for the next
// page ("" when the listing is exhausted). Pass the returned cursor back in
// SearchOptions.Cursor to continue. The Hub paginates via an RFC 5988 Link header.
func (c *Client) SearchPage(ctx context.Context, opts SearchOptions) ([]ModelInfo, string, error) {
	u := c.baseURL + "/api/models"
	if q := opts.values().Encode(); q != "" {
		u += "?" + q
	}
	var out []ModelInfo
	h, err := c.getJSONResp(ctx, u, &out)
	if err != nil {
		return nil, "", err
	}
	return out, nextCursorFromLink(h.Get("Link")), nil
}

// nextCursorFromLink extracts the cursor query param of the rel="next" URL in an RFC 5988
// Link header, or "" if there is no next page.
func nextCursorFromLink(link string) string {
	for part := range strings.SplitSeq(link, ",") {
		seg := strings.TrimSpace(part)
		if !strings.Contains(seg, `rel="next"`) {
			continue
		}
		lo, hi := strings.IndexByte(seg, '<'), strings.IndexByte(seg, '>')
		if lo < 0 || hi <= lo {
			return ""
		}
		u, err := url.Parse(seg[lo+1 : hi])
		if err != nil {
			return ""
		}
		return u.Query().Get("cursor")
	}
	return ""
}

// Model returns a single model by id (GET /api/models/{id}), including its safetensors
// parameter counts when available. Returns ErrNotFound if the id does not exist.
func (c *Client) Model(ctx context.Context, id string) (ModelInfo, error) {
	// The id may contain a slash (owner/name); that's a valid path and needs no escaping.
	u := c.baseURL + "/api/models/" + strings.TrimPrefix(id, "/")
	var out ModelInfo
	if err := c.getJSON(ctx, u, &out); err != nil {
		return ModelInfo{}, err
	}
	return out, nil
}

// getJSON performs a GET and decodes a JSON body, mapping notable status codes to
// sentinel errors.
func (c *Client) getJSON(ctx context.Context, url string, dst any) error {
	_, err := c.getJSONResp(ctx, url, dst)
	return err
}

// getJSONResp is getJSON that also returns the response header, so callers that need
// pagination metadata (the Link header) can read it.
func (c *Client) getJSONResp(ctx context.Context, url string, dst any) (http.Header, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("huggingface: build request: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", c.userAgent)

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("huggingface: request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	switch {
	case resp.StatusCode == http.StatusNotFound:
		return nil, ErrNotFound
	case resp.StatusCode == http.StatusTooManyRequests:
		return nil, ErrRateLimited
	case resp.StatusCode < 200 || resp.StatusCode >= 300:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("huggingface: unexpected status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	if err := json.NewDecoder(resp.Body).Decode(dst); err != nil {
		return nil, fmt.Errorf("huggingface: decode response: %w", err)
	}
	return resp.Header, nil
}
