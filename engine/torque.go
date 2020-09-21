package engine

import (
	"reflect"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Alpha                            = NewScalarParam("alpha", "", "Landau-Lifshitz damping constant")
	Xi                               = NewScalarParam("xi", "", "Non-adiabaticity of spin-transfer-torque")
	Pol                              = NewScalarParam("Pol", "", "Electrical current polarization")
	Lambda                           = NewScalarParam("Lambda", "", "Slonczewski Λ parameter")
	EpsilonPrime                     = NewScalarParam("EpsilonPrime", "", "Slonczewski secondairy STT term ε'")
	FrozenSpins                      = NewScalarParam("frozenspins", "", "Defines spins that should be fixed") // 1 - frozen, 0 - free. TODO: check if it only contains 0/1 values
	FreeLayerThickness               = NewScalarParam("FreeLayerThickness", "m", "Slonczewski free layer thickness (if set to zero (default), then the thickness will be deduced from the mesh size)")
	SOThickness                      = NewScalarParam("SOThickness", "m", "SOT freelayer thickness") // 自己加的
	HFLoverHDL                       = NewScalarParam("HFLoverHDL", "", "SOT HFL/HDL")               // 自己加的
	SPINHALL                         = NewScalarParam("SPINHALL", "", "SOT spin hall angle")         // 自己加的
	FixedLayer                       = NewExcitation("FixedLayer", "", "Slonczewski fixed layer polarization")
	SOTFixedLayer                    = NewExcitation("SOTFixedLayer", "", "SOT sigma") // 自己加的
	Torque                           = NewVectorField("torque", "T", "Total torque/γ0", SetTorque)
	LLTorque                         = NewVectorField("LLtorque", "T", "Landau-Lifshitz torque/γ0", SetLLTorque)
	STTorque                         = NewVectorField("STTorque", "T", "Spin-transfer torque/γ0", AddSTTorque)
	SOTorque                         = NewVectorField("SOTorque", "T", "Spin-orbit torque/γ0", AddSOTorque) // 自己加的
	J                                = NewExcitation("J", "A/m2", "Electrical current density")
	JSOT                             = NewExcitation("JSOT", "A/m2", "SOT Electrical current density") // 自己加的
	MaxTorque                        = NewScalarValue("maxTorque", "T", "Maximum torque/γ0, over all cells", GetMaxTorque)
	GammaLL                  float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Precess                          = true
	DisableZhangLiTorque             = false
	DisableSlonczewskiTorque         = false
	DisableSOTorque                  = false          // 自己加的
	fixedLayerPosition               = FIXEDLAYER_TOP // instructs mumax3 how free and fixed layers are stacked along +z direction
)

func init() {
	Pol.setUniform([]float64{1}) // default spin polarization
	Lambda.Set(1)                // sensible default value (?).
	DeclVar("GammaLL", &GammaLL, "Gyromagnetic ratio in rad/Ts")
	DeclVar("DisableZhangLiTorque", &DisableZhangLiTorque, "Disables Zhang-Li torque (default=false)")
	DeclVar("DisableSlonczewskiTorque", &DisableSlonczewskiTorque, "Disables Slonczewski torque (default=false)")
	DeclVar("DoPrecess", &Precess, "Enables LL precession (default=true)")
	DeclLValue("FixedLayerPosition", &flposition{}, "Position of the fixed layer: FIXEDLAYER_TOP, FIXEDLAYER_BOTTOM (default=FIXEDLAYER_TOP)")
	DeclROnly("FIXEDLAYER_TOP", FIXEDLAYER_TOP, "FixedLayerPosition = FIXEDLAYER_TOP instructs mumax3 that fixed layer is on top of the free layer")
	DeclROnly("FIXEDLAYER_BOTTOM", FIXEDLAYER_BOTTOM, "FixedLayerPosition = FIXEDLAYER_BOTTOM instructs mumax3 that fixed layer is underneath of the free layer")
}

// Sets dst to the current total torque
func SetTorque(dst *data.Slice) {
	SetLLTorque(dst)
	AddSTTorque(dst)
	AddSOTorque(dst) // 自己加的
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorque(dst *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorque(dst *data.Slice) {
	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	if !DisableZhangLiTorque {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		xi := Xi.MSlice()
		defer xi.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		cuda.AddZhangLiTorque(dst, M.Buffer(), msat, j, alpha, xi, pol, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
		msat := Msat.MSlice()
		defer msat.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		fixedP := FixedLayer.MSlice()
		defer fixedP.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		epsPrime := EpsilonPrime.MSlice()
		defer epsPrime.Recycle()
		thickness := FreeLayerThickness.MSlice()
		defer thickness.Recycle()
		cuda.AddSlonczewskiTorque2(dst, M.Buffer(),
			msat, j, fixedP, alpha, pol, lambda, epsPrime,
			thickness,
			CurrentSignFromFixedLayerPosition[fixedLayerPosition],
			Mesh())
	}
}

func FreezeSpins(dst *data.Slice) {
	if !FrozenSpins.isZero() {
		cuda.ZeroMask(dst, FrozenSpins.gpuLUT1(), regions.Gpu())
	}
}

func GetMaxTorque() float64 {
	torque := ValueOf(Torque)
	defer cuda.Recycle(torque)
	return cuda.MaxVecNorm(torque)
}

type FixedLayerPosition int

const (
	FIXEDLAYER_TOP FixedLayerPosition = iota + 1
	FIXEDLAYER_BOTTOM
)

var (
	CurrentSignFromFixedLayerPosition = map[FixedLayerPosition]float64{
		FIXEDLAYER_TOP:    1.0,
		FIXEDLAYER_BOTTOM: -1.0,
	}
)

type flposition struct{}

func (*flposition) Eval() interface{} { return fixedLayerPosition }
func (*flposition) SetValue(v interface{}) {
	drainOutput()
	fixedLayerPosition = v.(FixedLayerPosition)
}
func (*flposition) Type() reflect.Type { return reflect.TypeOf(FixedLayerPosition(FIXEDLAYER_TOP)) }

// 自己加的
func AddSOTorque(dst *data.Slice) {
	if JSOT.isZero() {
		return
	}
	util.AssertMsg(!SPINHALL.isZero(), "spin hall angle should not be 0")
	jsotspin, sotrec := J.Slice()
	if sotrec {
		defer cuda.Recycle(jsotspin)
	}
	sotfl, sotrec := SOTFixedLayer.Slice()
	if sotrec {
		defer cuda.Recycle(sotfl)
	}
	if !DisableSOTorque {
		msat := Msat.MSlice()
		defer msat.Recycle()
		jsot := JSOT.MSlice()
		defer jsot.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		sotfixedP := SOTFixedLayer.MSlice()
		defer sotfixedP.Recycle()
		spinhall := SPINHALL.MSlice()
		defer spinhall.Recycle()
		hfloverhdl := HFLoverHDL.MSlice()
		defer hfloverhdl.Recycle()
		sothickness := SOThickness.MSlice()
		defer sothickness.Recycle()
		cuda.AddSOTorque(dst, M.Buffer(),
			msat, jsot, sotfixedP, alpha, spinhall, hfloverhdl,
			sothickness, Mesh())
	}
}
