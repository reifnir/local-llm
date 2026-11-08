# `make help [target]`, read from the Makefile itself. Public targets are documented as:
#
#   ##@ Group              a heading in the `make help` list; targets below it belong to it
#   ## Usage:   ...        detail lines (shown by `make help <target>`)
#   target: ## summary     the one-liner (shown by `make help`)
#
# usage: awk [-v topic=<target>] -f scripts/make-help.awk Makefile

/^##@ / { if (topic == "") printf "\n%s\n", substr($0, 5); next }

/^##( |$)/ { doc = doc substr($0, 4) "\n"; next }

/^[a-z][a-z0-9-]*:.*## / {
    name = $0; sub(/:.*/, "", name)
    summary = $0; sub(/^[^#]*## /, "", summary)
    names = names " " name
    if (topic == "") {
        printf "  %-16s %s\n", name, summary
    } else if (name == topic) {
        printf "make %s: %s\n\n%s", name, summary, doc
        found = 1
    }
}

{ doc = "" }

END {
    if (topic != "" && !found) {
        printf "make help: no target '%s'. Targets:%s\n", topic, names > "/dev/stderr"
        exit 1
    }
}
