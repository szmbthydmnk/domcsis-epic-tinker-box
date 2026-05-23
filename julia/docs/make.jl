using Documenter
using DomcsisEpicTinkerBox

Documenter.makedocs(
    sitename = "DomcsisEpicTinkerBox.jl",
    modules  = [DomcsisEpicTinkerBox],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "API"  => "api.md",
    ],
)

Documenter.deploydocs(
    repo   = "github.com/szmbthydmnk/domcsis-epic-tinker-box.git",
    branch = "gh-pages",
    devbranch = "main",
)
