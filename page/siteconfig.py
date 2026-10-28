"""Hosting and branding for one deployment of the Binding Atlas, read from a small TOML file (config/*.toml).

Every value is optional; the empty default (config/default.toml) builds a page with relative URLs only, no canonical
link and no site name, servable from any path prefix. The template carries placeholders for these values; a line
whose placeholder has no value is dropped (so an unset canonical URL removes <link rel="canonical"> and og:url).

    [site]
    name = "example.org"                       # og:site_name
    base_url = "https://example.org/atlas/"    # absolute URL of the directory the page is served from
    clean_urls = true                          # canonical = base_url + "binding-atlas" (else "binding-atlas.html")
    card_label = ""                            # URL text printed on the link card (default: canonical without scheme)
    [nav]
    back_url = "https://example.org/"          # optional back-link in the page byline
    back_label = "example.org"
    [font]                                     # optional host-supplied web font (not shipped in this repository)
    name = "Some Typewriter"                   # its name, for the CSS comment
    family = "Some TW"                         # CSS family alias, first in the page's font stacks
    url = "fonts/"                             # URL prefix of the files, relative to the page or absolute
    stem = "SomeTW"                            # files: {stem}-Regular / -Bold / -Italic / -BoldItalic .woff2
    meta = "Some Typewriter; fallback Courier Prime (OFL)"   # <meta name="typeface">
    credit = "Set in Some Typewriter with Courier Prime."     # footer line
Without [font] the page is set in Courier Prime (OFL, from Google Fonts), falling back to the system monospace.
"""
from __future__ import annotations

import html
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PAGE = "binding-atlas.html"
CLEAN = "binding-atlas"
MARKDOWN = "binding-atlas.md"
CARD = "binding-atlas-card.png"
PLACEHOLDERS = ("__CANONICAL_URL__", "__SITE_NAME__", "__CARD_URL__", "__BACK_LINK__", "__FONT_FACES__", "__FONT__",
                "__FONT_META__", "__FONT_CREDIT__")
INLINE = ("__BACK_LINK__", "__FONT__")          # may render empty without dropping their line
FACES = (("Regular", 400, "normal"), ("Bold", 700, "normal"), ("Italic", 400, "italic"), ("BoldItalic", 700, "italic"))
DEFAULT_FONT_META = "Courier Prime (OFL)"
DEFAULT_FONT_CREDIT = "Set in Courier Prime."
_CSS_SAFE = re.compile(r"^[\w .:/@%+-]*$")      # values that go into CSS strings / url()


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class SiteConfig:
    name: str = ""
    base_url: str = ""
    clean_urls: bool = True
    card_label: str = ""
    back_url: str = ""
    back_label: str = ""
    font_name: str = ""
    font_family: str = ""
    font_url: str = ""
    font_stem: str = ""
    font_meta: str = ""
    font_credit: str = ""

    def __post_init__(self) -> None:
        for k in ("font_name", "font_family", "font_url", "font_stem"):
            if not _CSS_SAFE.match(getattr(self, k)):
                raise ConfigError(f"font.{k[5:]} has characters not allowed in CSS: {getattr(self, k)!r}")
        if self.font_family and not self.font_stem:
            raise ConfigError("font.family needs font.stem (the woff2 file stem)")
        if self.base_url:
            u = urlparse(self.base_url)
            if u.scheme not in ("http", "https") or not u.netloc or not self.base_url.endswith("/"):
                raise ConfigError(f"site.base_url must be an absolute http(s) URL ending in '/': {self.base_url!r}")
        if bool(self.back_url) != bool(self.back_label):
            raise ConfigError("nav.back_url and nav.back_label are set together or not at all")

    @classmethod
    def load(cls, path: Path | None) -> "SiteConfig":
        if path is None:
            return cls()
        try:
            raw = tomllib.loads(Path(path).read_text())
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError(f"cannot read site config {path}: {e}") from e
        site, nav, font = raw.pop("site", {}), raw.pop("nav", {}), raw.pop("font", {})
        if raw:
            raise ConfigError(f"{path}: unknown sections {sorted(raw)} (expected [site], [nav], [font])")
        known_site, known_nav = {"name", "base_url", "clean_urls", "card_label"}, {"back_url", "back_label"}
        known_font = {"name", "family", "url", "stem", "meta", "credit"}
        bad = sorted(set(site) - known_site) + sorted(set(nav) - known_nav) + sorted(set(font) - known_font)
        if bad:
            raise ConfigError(f"{path}: unknown keys {bad}")
        for k, v in (site | nav | {f"font.{k}": v for k, v in font.items()}).items():
            if not isinstance(v, bool if k == "clean_urls" else str):
                raise ConfigError(f"{path}: {k} has the wrong type ({type(v).__name__})")
        return cls(**site, **nav, **{f"font_{k}": v for k, v in font.items()})

    # ---- derived URLs
    @property
    def canonical_url(self) -> str:
        """Absolute URL of the page, or "" when the deployment has no base URL."""
        return self.base_url + (CLEAN if self.clean_urls else PAGE) if self.base_url else ""

    @property
    def card_url(self) -> str:
        """og:image / twitter:image: absolute when a base URL is set, else relative to the page."""
        return self.base_url + CARD if self.base_url else CARD

    @property
    def page_link(self) -> str:
        """Where the Markdown twin points back to: the canonical URL, else the page file next to it."""
        return self.canonical_url or PAGE

    @property
    def card_text(self) -> str:
        if self.card_label:
            return self.card_label
        c = self.canonical_url
        return c.split("://", 1)[1] if c else ""

    def font_faces(self, url: str | None = None) -> list[str]:
        """@font-face rules for the configured web font (none without one); `url` overrides the file prefix."""
        if not self.font_family:
            return []
        u = self.font_url if url is None else url
        return [f'@font-face {{ font-family: "{self.font_family}"; src: url("{u}{self.font_stem}-{style}.woff2") '
                f'format("woff2"); font-weight: {w}; font-style: {fs}; font-display: swap; }}' for style, w, fs in FACES]

    def placeholder_values(self) -> dict[str, str]:
        back = (f'<a href="{html.escape(self.back_url)}">{html.escape(self.back_label)}</a>' if self.back_url else "")
        faces = ""
        if self.font_family:
            where = "self-hosted next to the page" if "://" not in self.font_url and not self.font_url.startswith("/") \
                else f"from {self.font_url}"
            faces = "\n".join([f"/* {self.font_name or self.font_family} ({where}); Courier Prime fills the glyphs it "
                                f"lacks (→ × ≈ ≥ “ ” …). */"] + self.font_faces())
        return {"__CANONICAL_URL__": html.escape(self.canonical_url), "__SITE_NAME__": html.escape(self.name),
                "__CARD_URL__": html.escape(self.card_url), "__BACK_LINK__": back, "__FONT_FACES__": faces,
                "__FONT__": f'"{self.font_family}", ' if self.font_family else "",
                "__FONT_META__": html.escape(self.font_meta or DEFAULT_FONT_META, quote=False).replace('"', "&quot;"),
                "__FONT_CREDIT__": html.escape(self.font_credit or DEFAULT_FONT_CREDIT, quote=False)}


def render_hosting(template: str, cfg: SiteConfig) -> str:
    """Fill the hosting placeholders. A line holding a placeholder whose value is empty is dropped, except for the
    inline ones (__BACK_LINK__, __FONT__), which simply vanish."""
    vals = cfg.placeholder_values()
    out: list[str] = []
    for line in template.splitlines(keepends=True):
        drop = False
        for ph in PLACEHOLDERS:
            if ph in line:
                if not vals[ph] and ph not in INLINE:
                    drop = True
                    break
                line = line.replace(ph, vals[ph])
        if not drop:
            out.append(line)
    return "".join(out)
