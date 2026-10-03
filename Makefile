GOCMD = go
GOBUILD = $(GOCMD) build -v
GOVET = $(GOCMD) vet
GOTEST = $(GOCMD) test -v

BIN_DIR = bin
TOOLS_BIN_DIR := $(abspath $(BIN_DIR))
GO_INSTALL := ./scripts/go_install.sh

NAME = blizzard
CMD = $(BIN_DIR)/$(NAME)
CMD_WINDOWS = $(BIN_DIR)/$(NAME).exe
PKG = ./cmd

TAG ?= $(shell git describe --tags 2>/dev/null || echo "dev")
COMMIT ?= $(shell git describe --always 2>/dev/null || echo "unknown")
BUILD_DATE ?= $(shell date -u +%m/%d/%Y)

LD_FLAGS = -X main.Version=$(TAG) -X main.GitCommit=$(COMMIT) -X main.Date=$(BUILD_DATE)

GOIMPORTS_BIN = goimports
GOIMPORTS_PKG = golang.org/x/tools/cmd/goimports
GOIMPORTS_VER = v0.33.0
GOIMPORTS := $(abspath $(TOOLS_BIN_DIR)/$(GOIMPORTS_BIN)-$(GOIMPORTS_VER))

all: fmt cmd cmd-windows

all-windows: fmt cmd-windows

build: cmd-windows

cmd: export GOOS := linux
cmd: export GOARCH := amd64
cmd:
	$(GOBUILD) -ldflags="$(LD_FLAGS)" -o $(CMD) $(PKG)

cmd-windows: export GOOS := windows
cmd-windows: export GOARCH := amd64
cmd-windows:
	$(GOBUILD) -ldflags="$(LD_FLAGS)" -o $(CMD_WINDOWS) $(PKG)

vet:
	$(GOVET) ./...

fmt: goimports
	$(GOIMPORTS) -w .

test:
	$(GOTEST) ./...

goimports:
	GOBIN=$(TOOLS_BIN_DIR) $(GO_INSTALL) $(GOIMPORTS_PKG) $(GOIMPORTS_BIN) $(GOIMPORTS_VER)

clean:
	rm -rf $(BIN_DIR)

.PHONY: cmd cmd-windows all all-windows vet fmt test goimports clean build