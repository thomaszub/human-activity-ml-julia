# Own RNN

using Flux: @functor, gate, Recur

mutable struct ORNNCell{A,V,S,T}
  Wi::A
  Wh::A
  b::V
  c::V
  σ_c::S
  σ_h::T
end

function ORNNCell(in::Integer, out::Integer;
                  init = Flux.glorot_uniform,
                  initb = Flux.zeros,
                  σ_c = relu,
                  σ_h = tanh)
  cell = ORNNCell(
          init(out * 3, in),
          init(out * 3, out),
          init(out * 3),
          initb(out),
          σ_c,
          σ_h
  )
  cell.b[gate(out, 1)] .= 1
  return cell
end

function (m::ORNNCell)(c, x)
  o = size(c, 1)
  g = m.Wi*x .+ m.Wh*c .+ m.b
  forget = σ.(gate(g, o, 1))
  cell = m.σ_c.(gate(g, o, 2))
  c = forget .* c .- (forget .- 1) .* cell
  h = m.σ_h.(gate(g, o, 3))
  return c, h
end

Flux.hidden(m::ORNNCell) = m.c

@functor ORNNCell

Base.show(io::IO, l::ORNNCell) =
  print(io, "ORNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ", ", l.σ_c , ", ", l.σ_h, ")")

ORNN(a...; ka...) = Recur(ORNNCell(a...; ka...))
