# Own RNN

using Flux: @functor, gate, Recur

Flux.gate(h, n, size) = (1:size) .+ h*(n-1)
Flux.gate(x::AbstractVector, h, n, size) = @view x[gate(h,n,size)]
Flux.gate(x::AbstractMatrix, h, n, size) = x[gate(h,n,size),:]

mutable struct ORNNCell{A,V,S,T}
  Wx::A
  Wc::A
  b::V
  c::V
  σ_c::S
  σ_h::T
end

function ORNNCell(in::Integer, cell::Integer, out::Integer;
                  init = Flux.glorot_uniform,
                  init_cell = Flux.zeros,
                  σ_c = relu,
                  σ_h = tanh)
  agg_size = 2 * cell + out
  cell = ORNNCell(
          init(agg_size, in),
          init(agg_size, cell),
          init(agg_size),
          init_cell(cell),
          σ_c,
          σ_h
  )
  cell.b[gate(out, 1)] .= 1
  return cell
end

function (m::ORNNCell)(c, x)
  o_c = size(m.Wc, 2)
  o_h = size(m.Wc, 1) - 2 * o_c
  g = m.Wx*x .+ m.Wc*c .+ m.b
  forget = σ.(gate(g, o_c, 1))
  cell = m.σ_c.(gate(g, o_c, 2))
  c = forget .* c .- (forget .- 1) .* cell
  h = m.σ_h.(gate(g, o_c, 3, o_h))
  return c, h
end

Flux.hidden(m::ORNNCell) = m.c

@functor ORNNCell

Base.show(io::IO, l::ORNNCell) =
  print(io, "ORNNCell(", size(l.Wx, 2), ", ", size(l.Wc, 2), ", ", size(l.Wc, 1) - 2 * size(l.Wc, 2), ", ", l.σ_c , ", ", l.σ_h, ")")

ORNN(a...; ka...) = Recur(ORNNCell(a...; ka...))
