SHELL := /usr/bin/env bash

PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
DESTDIR ?=
NATIVE_BIN := air_llm/build/native/airllm

.PHONY: native-launcher native-standalone install-native uninstall-native release-archive

native-launcher:
	@bash air_llm/scripts/build_native.sh

native-standalone:
	@bash air_llm/scripts/build_native_standalone.sh

install-native:
	@test -x "$(NATIVE_BIN)" || (echo "Build binary first: make native-standalone" >&2; exit 1)
	@install -d "$(DESTDIR)$(BINDIR)"
	@install -m 0755 "$(NATIVE_BIN)" "$(DESTDIR)$(BINDIR)/airllm"
	@echo "Installed $(DESTDIR)$(BINDIR)/airllm"

uninstall-native:
	@rm -f "$(DESTDIR)$(BINDIR)/airllm"
	@echo "Removed $(DESTDIR)$(BINDIR)/airllm"

release-archive:
	@test -n "$(VERSION)" || (echo "Set VERSION, e.g. make release-archive VERSION=v2.12.0 PLATFORM=linux-x86_64" >&2; exit 1)
	@test -n "$(PLATFORM)" || (echo "Set PLATFORM, e.g. PLATFORM=linux-x86_64" >&2; exit 1)
	@test -x "$(NATIVE_BIN)" || (echo "Build binary first: make native-standalone" >&2; exit 1)
	@mkdir -p dist/package
	@cp "$(NATIVE_BIN)" dist/package/airllm
	@cp LICENSE dist/package/LICENSE
	@cp Makefile dist/package/Makefile
	@cp air_llm/README.md dist/package/README.md
	@tar -C dist/package -czf "dist/airllm-$(VERSION)-$(PLATFORM).tar.gz" airllm LICENSE Makefile README.md
	@rm -rf dist/package
	@echo "Created dist/airllm-$(VERSION)-$(PLATFORM).tar.gz"
