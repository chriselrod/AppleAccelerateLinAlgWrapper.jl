module AppleAccelerateLinAlgWrapper

using libblastrampoline_jll, Libdl
using LinearAlgebra: BLAS, Adjoint, Transpose, StridedMatrix,
    UnitLowerTriangular, LowerTriangular, UnitUpperTriangular, UpperTriangular, AbstractTriangular
import LinearAlgebra

if Sys.isapple()
    default_blas_int() = Val(Int32)
else
    default_blas_int() = Val(LinearAlgebra.BlasInt)
end

const LIBBLASTRAMPOLINE_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

for f ∈ [:gemm, :getrf, :trsm]
    get = Symbol(:get_, f, :_addr)
    for (p, T) ∈ ((:s, :Float32), (:d, :Float64))
        pf = Symbol(p, f, :_)
        pf64 = Symbol(pf, Symbol("64_"))
        @eval begin
            function $get(::Val{$T}, ::Val{Int32})
                dlsym(LIBBLASTRAMPOLINE_HANDLE[], $(QuoteNode(pf)))::Ptr{Cvoid}
            end
            function $get(::Val{$T}, ::Val{Int64})
                dlsym(LIBBLASTRAMPOLINE_HANDLE[], $(QuoteNode(pf64)))::Ptr{Cvoid}
            end
        end
    end
end

function gemm!(transA::AbstractChar, transB::AbstractChar,
               alpha::Union{T, Bool},
               A::StridedMatrix{T}, B::StridedMatrix{T},
               beta::Union{T, Bool},
               C::StridedMatrix{T}, ::Val{BlasInt}) where {BlasInt <: Union{Int32,Int64},T<:Union{Float32,Float64}}
    m = size(A, transA == 'N' ? 1 : 2) % BlasInt
    ka = size(A, transA == 'N' ? 2 : 1) % BlasInt
    kb = size(B, transB == 'N' ? 1 : 2) % BlasInt
    n = size(B, transB == 'N' ? 2 : 1) % BlasInt
    if ka != kb || m != (size(C,1) % BlasInt) || n != (size(C,2) % BlasInt)
        throw(DimensionMismatch("A has size ($m,$ka), B has size ($kb,$n), C has size $(size(C))"))
    end
    fptr = get_gemm_addr(Val(T), Val(BlasInt))
#    ccall(:jl_breakpoint, Cvoid, (Any,), fptr)
    ccall(fptr, Cvoid, (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{T}, Ptr{T}, Ref{BlasInt},
                        Ptr{T}, Ref{BlasInt}, Ref{T}, Ptr{T},
                        Ref{BlasInt}, Clong, Clong),
          transA, transB, m, n,
          ka, alpha, A, max(1,stride(A,2)),
          B, max(1,stride(B,2)), beta, C,
          max(1,stride(C,2)) % BlasInt, 1, 1)
    C
end
function gemm(
    transA::AbstractChar, transB::AbstractChar, alpha::T, A::StridedMatrix{T}, B::StridedMatrix{T}, ::Val{BlasInt}
) where {BlasInt <: Union{Int32,Int64}, T <: Union{Float32,Float64}}
    gemm!(transA, transB, alpha, A, B, zero(T), similar(B, T, (size(A, transA == 'N' ? 1 : 2), size(B, transB == 'N' ? 2 : 1))), Val(BlasInt))
end

untranspose_flag(A::Adjoint, f::Bool = false) = untranspose_flag(parent(A), !f)
untranspose_flag(A::Transpose, f::Bool = false) = untranspose_flag(parent(A), !f)
untranspose_flag(A, f::Bool = false) = (A, f)


function gemm!(
    C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}, α = one(T), β = zero(T), ::Val{BlasInt} = default_blas_int()
) where {T<:Union{Float32,Float64},BlasInt}
    pA, fa = untranspose_flag(A)
    pB, fb = untranspose_flag(B)
    gemm!(ifelse(fa, 'T', 'N'), ifelse(fb, 'T', 'N'), α, parent(A), parent(B), β, C, Val(BlasInt))
end

struct LU{T,M<:StridedMatrix{T},BlasInt}
    factors::M
    ipiv::Vector{BlasInt}
    info::BlasInt
end

function getrf_ipiv(A::StridedMatrix, ::Val{BlasInt}) where {BlasInt}
    M,N = size(A)
    Vector{BlasInt}(undef, min(M,N))
end
function getrf!(
    A::StridedMatrix{T}, ipiv::StridedVector{BlasInt} = getrf_ipiv(A, default_blas_int())
) where {T <: Union{Float32,Float64}, BlasInt <: Union{Int32,Int64}}
    M, N = size(A)
    fptr = get_getrf_addr(Val(T), Val(BlasInt))
    info = Ref{BlasInt}()
    ccall(
        fptr, Cvoid, (Ref{BlasInt},Ref{BlasInt},Ptr{T},Ref{BlasInt},Ptr{BlasInt},Ref{BlasInt}),
        M % BlasInt, N % BlasInt, A, max(1,stride(A,2)) % BlasInt, ipiv, info
    )
    LU(A, ipiv, info[])
end
lu!(A::StridedMatrix, ::Val{BlasInt} = default_blas_int()) where {BlasInt} = getrf!(A, getrf_ipiv(A, Val(BlasInt)))
lu(A::StridedMatrix, ::Val{BlasInt} = default_blas_int()) where {BlasInt} = lu!(copy(A), Val(BlasInt))

uplochar(::Union{LowerTriangular,UnitLowerTriangular}) = 'L'
uplochar(::Union{UpperTriangular,UnitUpperTriangular}) = 'U'
diagchar(::Union{LowerTriangular,UpperTriangular}) = 'N'
diagchar(::Union{UnitLowerTriangular,UnitUpperTriangular}) = 'U'


# transpose wrappers should be external to the triangular wrappers of `trsm!`
function trsm!(
    B::StridedMatrix{T}, α::T, A::AbstractMatrix{T}, side::Char, ::Val{BlasInt} = default_blas_int()
) where {T<:Union{Float32,Float64},BlasInt<:Union{Int32,Int64}}
    M, N = size(A)
    pA, transa = untranspose_flag(A)
    uplo = uplochar(pA)
    diag = diagchar(pA)
    fptr = get_trsm_addr(Val(T), Val(BlasInt))
    ppA = parent(pA)
    ccall(
        fptr, Cvoid, (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ref{T}, Ptr{Float64}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt}),
        side, uplo, ifelse(transa, 'T', 'N'), diag, M % BlasInt, N % BlasInt, α, ppA, max(1,stride(ppA,2)) % BlasInt, B, max(1,stride(B,2)) % BlasInt
    )
    B
end

function rdiv!(
    A::StridedMatrix{T}, B::AbstractTriangular{T}, ::Val{BlasInt}
) where {T <: Union{Float32,Float64}, BlasInt <: Union{Int32,Int64}}
    trsm!(A, one(T), B, 'R', Val(BlasInt))
end
function ldiv!(
    A::StridedMatrix{T}, B::AbstractTriangular{T}, ::Val{BlasInt}
) where {T <: Union{Float32,Float64}, BlasInt <: Union{Int32,Int64}}
    trsm!(A, one(T), B, 'L', Val(BlasInt))
end


function _ipiv_cols!(A::LU, order::OrdinalRange, B::StridedVecOrMat)
    ipiv = A.ipiv
    @inbounds for i ∈ order
        i ≠ ipiv[i] && LinearAlgebra._swap_cols!(B, i, ipiv[i])
    end
    B
end
_apply_ipiv_rows!(A::LU, B::StridedVecOrMat) = _ipiv_rows!(A, 1 : length(A.ipiv), B)
_apply_inverse_ipiv_cols!(A::LU, B::StridedVecOrMat) = _ipiv_cols!(A, length(A.ipiv) : -1 : 1, B)
function rdiv!(A::StridedMatrix, B::LU{<:Any,<:StridedMatrix,BlasInt}) where {BlasInt}
    rdiv!(rdiv!(A, UpperTriangular(B.factors), Val(BlasInt)), UnitLowerTriangular(B.factors), Val(BlasInt))
    LinearAlgebra._apply_inverse_ipiv_cols!(B, A) # mutates `A`
end
function ldiv!(A::LU{<:Any,<:StridedMatrix,BlasInt}, B::StridedMatrix) where {BlasInt}
    _apply_ipiv_rows!(A, B)
    ldiv!(UpperTriangular(A.factors), ldiv!(UnitLowerTriangular(A.factors), B))
end

function rdiv!(A::StridedMatrix, B::StridedMatrix, ::Val{BlasInt} = default_blas_int()) where {BlasInt}
    rdiv!(A, lu!(B, Val(BlasInt)))
end
function ldiv!(A::StridedMatrix, B::StridedMatrix, ::Val{BlasInt} = default_blas_int()) where {BlasInt}
    ldiv!(lu!(A, Val(BlasInt)), B)
end

rdiv(A::StridedMatrix, B, ::Val{BlasInt} = default_blas_int()) where {BlasInt} = rdiv!(copy(A), copy(B), Val(BlasInt))
ldiv(A::StridedMatrix, B, ::Val{BlasInt} = default_blas_int()) where {BlasInt} = ldiv!(copy(A), copy(B), Val(BlasInt))

function __init__()
    LIBBLASTRAMPOLINE_HANDLE[] = libblastrampoline_jll.libblastrampoline_handle
    Sys.isapple() && BLAS.lbt_forward("/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate")
end

end # module
