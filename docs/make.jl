using SemanticTrajectories
using Documenter

DocMeta.setdocmeta!(SemanticTrajectories, :DocTestSetup, :(using SemanticTrajectories); recursive=true)

makedocs(;
    modules=[SemanticTrajectories],
    authors="fargolo <felipe.c.argolo@protonmail.com> and contributors",
    repo="https://github.com/fargolo/SemanticTrajectories.jl/blob/{commit}{path}#{line}",
    sitename="SemanticTrajectories.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://fargolo.github.io/SemanticTrajectories.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/fargolo/SemanticTrajectories.jl",
    devbranch="main",
)
