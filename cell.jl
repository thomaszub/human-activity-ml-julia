# Own RNN

using Flux: @functor, gate, Recur

mutable struct ORNNCell{A,V}
  Wi::A
  Wh::A
  b::V
  h::V
  c::V
end

function ORNNCell(in::Integer, out::Integer;
                  init = Flux.glorot_uniform, initb = Flux.zeros)
  cell = ORNNCell(
          init(out * 4, in),
          init(out * 4, out),
          init(out * 4),
          initb(out),
          initb(out)
  )
  cell.b[gate(out, 2)] .= 1
  return cell
end

function (m::ORNNCell)((h, c), x)
  b, o = m.b, size(h, 1)
  g = m.Wi*x .+ m.Wh*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  c = forget .* c .+ input .* cell
  h′ = output .* tanh.(c)
  return (h′, c), h′
end

Flux.hidden(m::ORNNCell) = (m.h, m.c)

@functor ORNNCell

Base.show(io::IO, l::ORNNCell) =
  print(io, "ORNNCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")

ORNN(a...; ka...) = Recur(ORNNCell(a...; ka...))
