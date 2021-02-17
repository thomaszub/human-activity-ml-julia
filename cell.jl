# Own RNN

using Flux: @functor, gate, Recur

mutable struct ORNNCell{A,V,T}
  Wi::A
  Wh::A
  b::V
  c::V
  act::T
end

function ORNNCell(in::Integer, out::Integer;
                  init = Flux.glorot_uniform,
                  initb = Flux.zeros,
                  activation = tanh)
  cell = ORNNCell(
          init(out * 3, in),
          init(out * 3, out),
          init(out * 3),
          initb(out),
          activation
  )
  cell.b[gate(out, 1)] .= 1
  return cell
end

function (m::ORNNCell)(c, x)
  o = size(c, 1)
  g = m.Wi*x .+ m.Wh*c .+ m.b
  forget = σ.(gate(g, o, 1))
  output = σ.(gate(g, o, 2))
  cell = m.act.(gate(g, o, 3))
  c = forget .* c .- (forget .- 1) .* cell
  h = output .* c
  return c, h
end

Flux.hidden(m::ORNNCell) = m.c

@functor ORNNCell

Base.show(io::IO, l::ORNNCell) =
  print(io, "ORNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷3, ", ", l.act ,")")

ORNN(a...; ka...) = Recur(ORNNCell(a...; ka...))
