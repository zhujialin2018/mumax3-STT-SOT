package cuda

import (
	"github.com/mumax/3/data"
)

// Add spin orbit torque to torque (Tesla).
// see sot.cu
func AddSOTorque(torque, m *data.Slice, Msat, JSOT, sotfixedP, alpha, spinhall, hfloverhdl, sothickness MSlice, mesh *data.Mesh) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_addsotorque_async(
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		JSOT.DevPtr(Z), JSOT.Mul(Z),
		sotfixedP.DevPtr(X), sotfixedP.Mul(X),
		sotfixedP.DevPtr(Y), sotfixedP.Mul(Y),
		sotfixedP.DevPtr(Z), sotfixedP.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		spinhall.DevPtr(0), spinhall.Mul(0),
		hfloverhdl.DevPtr(0), hfloverhdl.Mul(0),
		sothickness.DevPtr(0), sothickness.Mul(0),
		N, cfg)
}
