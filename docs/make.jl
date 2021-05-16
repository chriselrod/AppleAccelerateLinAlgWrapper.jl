using AppleAccelerateLinAlgWrapper
using Documenter

DocMeta.setdocmeta!(AppleAccelerateLinAlgWrapper, :DocTestSetup, :(using AppleAccelerateLinAlgWrapper); recursive=true)

makedocs(;
    modules=[AppleAccelerateLinAlgWrapper],
    authors="Chris Elrod <elrodc@gmail.com>, Elliot Saba, and contributors.",
    repo="https://github.com/"chriselrod"/AppleAccelerateLinAlgWrapper.jl/blob/{commit}{path}#{line}",
    sitename="AppleAccelerateLinAlgWrapper.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://"chriselrod".github.io/AppleAccelerateLinAlgWrapper.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/"chriselrod"/AppleAccelerateLinAlgWrapper.jl",
)
