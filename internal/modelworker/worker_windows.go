//go:build windows

package modelworker

import (
	"fmt"
	"os/exec"
	"sync"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	jobOnce   sync.Once
	jobHandle windows.Handle
	jobErr    error
)

func configureProc(cmd *exec.Cmd) {}

// afterStart puts the child in a job object that kills it when this process exits,
// so a crashed server cannot orphan a worker still holding GPU memory.
func afterStart(cmd *exec.Cmd) error {
	job, err := ensureJob()
	if err != nil {
		return err
	}

	handle, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE,
		false,
		uint32(cmd.Process.Pid),
	)
	if err != nil {
		return fmt.Errorf("modelworker: open child process: %w", err)
	}
	defer windows.CloseHandle(handle)

	if err := windows.AssignProcessToJobObject(job, handle); err != nil {
		return fmt.Errorf("modelworker: assign to job object: %w", err)
	}

	return nil
}

func ensureJob() (windows.Handle, error) {
	jobOnce.Do(func() {
		handle, err := windows.CreateJobObject(nil, nil)
		if err != nil {
			jobErr = fmt.Errorf("modelworker: create job object: %w", err)
			return
		}

		info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{
			BasicLimitInformation: windows.JOBOBJECT_BASIC_LIMIT_INFORMATION{
				LimitFlags: windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
			},
		}
		if _, err := windows.SetInformationJobObject(
			handle,
			windows.JobObjectExtendedLimitInformation,
			uintptr(unsafe.Pointer(&info)),
			uint32(unsafe.Sizeof(info)),
		); err != nil {
			windows.CloseHandle(handle)
			jobErr = fmt.Errorf("modelworker: set job object limits: %w", err)
			return
		}

		jobHandle = handle // deliberately never closed; the OS closes it on exit and kills the children
	})

	return jobHandle, jobErr
}
