//go:build linux

package modelworker

import (
	"os/exec"
	"syscall"
)

// Pdeathsig kills the child if this process dies, backing up the stdin-EOF guard.
func configureProc(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
}

func afterStart(cmd *exec.Cmd) error { return nil }
