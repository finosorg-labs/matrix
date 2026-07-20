//go:build lib

// Prebuilt mode: link against prebuilt static library.
// Usage: go build -tags lib
// Requires: prebuilt libfinkit_matrix_static.a in build/lib/
package matrix

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo CFLAGS: -I${SRCDIR}/modules/platform/include
#cgo linux,amd64   LDFLAGS: -L${SRCDIR}/build/linux_amd64 -lfinkit_platform_static -lm -lgfortran -lpthread
#cgo linux,arm64   LDFLAGS: -L${SRCDIR}/build/linux_arm64 -lfinkit_platform_static -lm -lgfortran -lpthread
#cgo darwin,amd64  LDFLAGS: -L${SRCDIR}/build/darwin_amd64 -lfinkit_platform_static -lm -lgfortran -lpthread
#cgo darwin,arm64  LDFLAGS: -L${SRCDIR}/build/darwin_arm64 -lfinkit_platform_static -lm -lgfortran -lpthread
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build/windows_amd64 -lfinkit_platform_static -lm -lgfortran -lpthread

#cgo linux,amd64   LDFLAGS: -L${SRCDIR}/build/linux_amd64 -lfinkit_matrix_static -lm -lgfortran -lpthread
#cgo linux,arm64   LDFLAGS: -L${SRCDIR}/build/linux_arm64 -lfinkit_matrix_static -lm -lgfortran -lpthread
#cgo darwin,amd64  LDFLAGS: -L${SRCDIR}/build/darwin_amd64 -lfinkit_matrix_static -lm -lgfortran -lpthread
#cgo darwin,arm64  LDFLAGS: -L${SRCDIR}/build/darwin_arm64 -lfinkit_matrix_static -lm -lgfortran -lpthread
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/build/windows_amd64 -lfinkit_matrix_static -lm -lgfortran -lpthread

// Forward declarations for fc_init/fc_cleanup (no public header)
int fc_init(void);
void fc_cleanup(void);
*/
import "C"

func init() {
	C.fc_init()
}
