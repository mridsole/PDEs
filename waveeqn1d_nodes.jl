using PyPlot

type WaveWire

  u::Array{Float64, 2}
  t::Float64
  t_n::Int

  c::Array{Float64, 1}   # wave speed
  d::Array{Float64, 1}   # damping

  dx::Float64
  dt::Float64

  # temporary intermediate solution storage
  u_x::Array{Float64, 1}
  u_t::Array{Float64, 1}
  u_xx::Array{Float64, 1}

  # for indexing previous time steps - 3 is newest
  t_n1::Int
  t_n2::Int
  t_n3::Int

  bc_1::Function
  bc_2::Function

  # inner constructor
  function WaveWire(n_x, dx, dt, c, d, bc_1, bc_2)
    return new(zeros(Float64, (n_x, 3)), 0.0, 0, c, d, dx, dt, zeros(Float64, (n_x - 1)),
      zeros(Float64, (n_x - 1)), zeros(Float64, (n_x - 2)), 1, 2, 3, bc_1, bc_2)
  end
end

type WaveWireNode

  # each pair stores a wire, and which end is connected to this node
  # false = start (index 1), true = end (index end)
  connects::Array{Pair{WaveWire, Bool}, 1}

  # inner constructor
  function WaveWireNode()
    return new(Array{Pair{WaveWire, Bool}, 1}())
  end
end

function step!(soln::WaveWire)

  n_x, _n_t = size(soln.u)
  t_n1 = soln.t_n1; t_n2 = soln.t_n2; t_n3 = soln.t_n3;

  # compute first spatial derivative
  for i = 1:(n_x - 1)
    soln.u_x[i] = (soln.u[i + 1, t_n2] - soln.u[i, t_n2]) / soln.dx
  end

  # compuate second spatial derivative
  for i = 1:(n_x - 2)
    soln.u_xx[i] = (soln.u_x[i + 1] - soln.u_x[i]) / soln.dx
  end

  # compuate first temporal derivative
  for i = 1:n_x
    soln.u_t[i] = (soln.u[i, t_n2] - soln.u[i, t_n1]) / soln.dt
  end

  # update non-boundary bits
  for i = 2:(n_x - 1)
    u_tt = soln.c[i]^2 * soln.u_xx[i - 1]
    soln.u[i, t_n3] = soln.dt^2 * (u_tt - soln.d[i] * soln.u_t[i]) + 2 * soln.u[i, t_n2] - soln.u[i, t_n1]
  end

  # apply boundary conditions
  soln.u[1, t_n3] = soln.bc_1(soln)
  soln.u[n_x, t_n3] = soln.bc_2(soln)

  # apply sources

  soln.t_n += 1
  soln.t += soln.dt

  # switch indexing
  soln.t_n3 = t_n1
  soln.t_n1 = t_n2
  soln.t_n2 = t_n3

end

function init_solution!(soln::WaveWire, u_init::Array{Float64, 1},
  u_t_init::Array{Float64, 1})

  soln.t_n1 = 1; soln.t_n2 = 2; soln.t_n3 = 3;

  soln.u[:, soln.t_n1] = u_init
  soln.u[:, soln.t_n2] = u_init + u_t_init .* soln.dt

  # some other set up stuff
  soln.u_t = u_t_init
  soln.t_n = 2
  soln.t = soln.dt * soln.t_n
end

function main()

  n_x = 1000
  dx = 0.0001; dt = 0.00002
  c = 4.9 * ones(Float64, (n_x))

  x = collect(linspace(0, (n_x - 1) * dx, n_x))

  # make a little window thing

  d = diff(exp(-10000 * (x - (n_x * dx / 2)) .^ 2))
  d = [d; d[end]]

  d = 400000 * d / norm(d) .* reverse(x)

  plot(x, d)
  pause(2)
  figure()

  # let's try zero BCs
  # bc1(soln::WaveWire) =  round(soln.t % 0.02 / 0.02)
  # bc1(soln::WaveWire) = sin(1000 * soln.t) # + 0.5 * sin(2200 * soln.t)
  # bc1(soln::WaveWire) = soln.c[1] * soln.u_x[1] * soln.dt + soln.u[1, soln.t_n2]
  bc2(soln::WaveWire) = -soln.c[end] * soln.u_x[end] * soln.dt + soln.u[end, soln.t_n2]

  bc1(soln::WaveWire) = 0
  # bc2(soln::WaveWire) = 0

  # create solution storage
  soln = WaveWire(n_x, dx, dt, c, d, bc1, bc2)

  # initialize solution with initial conditions
  u_init = zeros(n_x)
  u_t_init = zeros(n_x)

  # u_init[201:250] = linspace(0, 2, 50) .^ 2
  # u_init[251:280] = linspace(2, 0, 30) .^ 2

  z = collect(linspace(0, 99 * dx, 100))
  u_init[301:400] = exp(-500000 * (z - 50 * dx) .^ 2)


  init_solution!(soln, u_init, u_t_init)

  # plot initial distribution
  plot(x, u_init)
  figure()

  # number of iterations
  n = 100000

  # plot the data
  for i = 1:n
    if i % 10 == 0
      cla()
      plot(soln.u[:, soln.t_n3])
      ylim(-2, 2)
      # pause(0.001)
      println(i)
    end

    step!(soln)
  end

end
