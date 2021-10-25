module Sch

using Distributed
import MemPool: DRef, poolset
import Statistics: mean

import ..Dagger
import ..Dagger: Context, Processor, StorageResource, StorageDevice, Thunk, WeakThunk, ThunkFuture, ThunkFailedException, Chunk, OSProc, AnyScope, CPURAMDevice
import ..Dagger: order, free!, dependents, noffspring, istask, inputs, unwrap_weak_checked, affinity, tochunk, timespan_start, timespan_finish, unrelease, procs, move, chunktype, processor, default_enabled, get_processors, get_storage_resources, get_storage_devices, get_parent, execute!, rmprocs!, addprocs!, thunk_processor, constrain, cputhreadtime, storage_resource, storage_available, storage_pressure, storage_capacity

const OneToMany = Dict{Thunk, Set{Thunk}}

include("util.jl")
include("fault-handler.jl")
include("dynamic.jl")

mutable struct ProcessorCacheEntry
    gproc::OSProc
    proc::Processor
    next::ProcessorCacheEntry

    ProcessorCacheEntry(gproc::OSProc, proc::Processor) = new(gproc, proc)
end
Base.isequal(p1::ProcessorCacheEntry, p2::ProcessorCacheEntry) =
    p1.proc === p2.proc
function Base.show(io::IO, entry::ProcessorCacheEntry)
    entries = 1
    next = entry.next
    while next !== entry
        entries += 1
        next = next.next
    end
    print(io, "ProcessorCacheEntry(pid $(entry.gproc.pid), $(entry.proc), $entries entries)")
end

"""
    ComputeState

The internal state-holding struct of the scheduler.

Fields:
- `uid::UInt64` - Unique identifier for this scheduler instance
- `waiting::OneToMany` - Map from downstream `Thunk` to upstream `Thunk`s that still need to execute
- `waiting_data::Dict{Union{Thunk,Chunk},Set{Thunk}}` - Map from input `Chunk`/upstream `Thunk` to all unfinished downstream `Thunk`s, to retain caches
- `ready::Vector{Thunk}` - The list of `Thunk`s that are ready to execute
- `cache::WeakKeyDict{Thunk, Any}` - Maps from a finished `Thunk` to it's cached result, often a DRef
- `running::Set{Thunk}` - The set of currently-running `Thunk`s
- `running_on::Dict{Thunk,OSProc}` - Map from `Thunk` to the OS process executing it
- `thunk_dict::Dict{Int, WeakThunk}` - Maps from thunk IDs to a `Thunk`
- `node_order::Any` - Function that returns the order of a thunk
- `worker_time_pressure::Dict{Int,Dict{Processor,UInt64}}` - Maps from worker ID to processor pressure
- `worker_storage_pressure::Dict{Int,Dict{StorageResource,UInt64}}` - Maps from worker ID to storage resource pressure
- `worker_storage_capacity::Dict{Int,Dict{StorageResource,UInt64}}` - Maps from worker ID to storage resource capacity
- `worker_loadavg::Dict{Int,NTuple{3,Float64}}` - Worker load average
- `worker_chans::Dict{Int, Tuple{RemoteChannel,RemoteChannel}}` - Communication channels between the scheduler and each worker
- `procs_cache_list::Base.RefValue{Union{ProcessorCacheEntry,Nothing}}` - Cached linked list of processors ready to be used
- `signature_time_cost::Dict{Type{<:Tuple},UInt64}` - Cache of estimated CPU time (in nanoseconds) required to compute calls with the given signature
- `signature_alloc_cost::Dict{Type{<:Tuple},UInt64}` - Cache of estimated CPU RAM (in bytes) required to compute calls with the given signature
- `transfer_rate::Ref{UInt64}` - Estimate of the network transfer rate in bytes per second
- `halt::Base.Event` - Event indicating that the scheduler is halting
- `lock::ReentrantLock` - Lock around operations which modify the state
- `futures::Dict{Thunk, Vector{ThunkFuture}}` - Futures registered for waiting on the result of a thunk.
- `errored::WeakKeyDict{Thunk,Bool}` - Indicates if a thunk's result is an error.
- `chan::RemoteChannel{Channel{Any}}` - Channel for receiving completed thunks.
"""
struct ComputeState
    uid::UInt64
    waiting::OneToMany
    waiting_data::Dict{Union{Thunk,Chunk},Set{Thunk}}
    ready::Vector{Thunk}
    cache::WeakKeyDict{Thunk, Any}
    running::Set{Thunk}
    running_on::Dict{Thunk,OSProc}
    thunk_dict::Dict{Int, WeakThunk}
    node_order::Any
    worker_time_pressure::Dict{Int,Dict{Processor,UInt64}}
    worker_storage_pressure::Dict{Int,Dict{StorageResource,UInt64}}
    worker_storage_capacity::Dict{Int,Dict{StorageResource,UInt64}}
    worker_loadavg::Dict{Int,NTuple{3,Float64}}
    worker_chans::Dict{Int, Tuple{RemoteChannel,RemoteChannel}}
    procs_cache_list::Base.RefValue{Union{ProcessorCacheEntry,Nothing}}
    signature_time_cost::Dict{Type{<:Tuple},UInt64}
    signature_alloc_cost::Dict{Type{<:Tuple},UInt64}
    transfer_rate::Ref{UInt64}
    halt::Base.Event
    lock::ReentrantLock
    futures::Dict{Thunk, Vector{ThunkFuture}}
    errored::WeakKeyDict{Thunk,Bool}
    chan::RemoteChannel{Channel{Any}}
end

function start_state(deps::Dict, node_order, chan)
    state = ComputeState(rand(UInt64),
                         OneToMany(),
                         deps,
                         Vector{Thunk}(undef, 0),
                         Dict{Thunk, Any}(),
                         Set{Thunk}(),
                         Dict{Thunk,OSProc}(),
                         Dict{Int, WeakThunk}(),
                         node_order,
                         Dict{Int,Dict{Processor,UInt64}}(),
                         Dict{Int,Dict{StorageResource,UInt64}}(),
                         Dict{Int,Dict{StorageResource,UInt64}}(),
                         Dict{Int,NTuple{3,Float64}}(),
                         Dict{Int, Tuple{RemoteChannel,RemoteChannel}}(),
                         Ref{Union{ProcessorCacheEntry,Nothing}}(nothing),
                         Dict{Type{<:Tuple},UInt64}(),
                         Dict{Type{<:Tuple},UInt64}(),
                         Ref{UInt64}(1_000_000),
                         Base.Event(),
                         ReentrantLock(),
                         Dict{Thunk, Vector{ThunkFuture}}(),
                         WeakKeyDict{Thunk,Bool}(),
                         chan)

    for k in sort(collect(keys(deps)), by=node_order)
        if istask(k)
            waiting = Set{Thunk}(Iterators.filter(istask, inputs(k)))
            if isempty(waiting)
                push!(state.ready, k)
            else
                state.waiting[k] = waiting
            end
        end
    end
    state
end

"""
    SchedulerOptions

Stores DAG-global options to be passed to the Dagger.Sch scheduler.

# Arguments
- `single::Int=0`: Force all work onto worker with specified id. `0` disables
this option.
- `proclist=nothing`: Force scheduler to use one or more processors that are
instances/subtypes of a contained type. Alternatively, a function can be
supplied, and the function will be called with a processor as the sole
argument and should return a `Bool` result to indicate whether or not to use
the given processor. `nothing` enables all default processors.
- `allow_errors::Bool=true`: Allow thunks to error without affecting
non-dependent thunks.
- `checkpoint=nothing`: If not `nothing`, uses the provided function to save
the final result of the current scheduler invocation to persistent storage, for
later retrieval by `restore`.
- `restore=nothing`: If not `nothing`, uses the provided function to return the
(cached) final result of the current scheduler invocation, were it to execute.
If this returns a `Chunk`, all thunks will be skipped, and the `Chunk` will be
returned.  If `nothing` is returned, restoring is skipped, and the scheduler
will execute as usual. If this function throws an error, restoring will be
skipped, and the error will be displayed.
- `round_robin::Bool=false`: Whether to schedule in round-robin mode, which
spreads load instead of the default behavior of filling processors to capacity.
"""
Base.@kwdef struct SchedulerOptions
    single::Int = 0
    proclist = nothing
    allow_errors::Bool = false
    checkpoint = nothing
    restore = nothing
    round_robin::Bool = false
end

"""
    ThunkOptions

Stores Thunk-local options to be passed to the Dagger.Sch scheduler.

# Arguments
- `single::Int=0`: Force thunk onto worker with specified id. `0` disables this
option.
- `proclist=nothing`: Force thunk to use one or more processors that are
instances/subtypes of a contained type. Alternatively, a function can be
supplied, and the function will be called with a processor as the sole
argument and should return a `Bool` result to indicate whether or not to use
the given processor. `nothing` enables all default processors.
- `procutil::Dict{Type,Any}=Dict{Type,Any}()`: Indicates the maximum expected
processor utilization for this thunk. Each keypair maps a processor type to
the utilization, where the value can be a real (approx. the number of processors
of this type utilized), or `MaxUtilization()` (utilizes all processors of this
type). By default, the scheduler assumes that this thunk only uses one
processor.
- `allow_errors::Bool=true`: Allow this thunk to error without affecting
non-dependent thunks.
- `checkpoint=nothing`: If not `nothing`, uses the provided function to save
the result of the thunk to persistent storage, for later retrieval by
`restore`.
- `restore=nothing`: If not `nothing`, uses the provided function to return the
(cached) result of this thunk, were it to execute.  If this returns a `Chunk`,
this thunk will be skipped, and its result will be set to the `Chunk`.  If
`nothing` is returned, restoring is skipped, and the thunk will execute as
usual. If this function throws an error, restoring will be skipped, and the
error will be displayed.
"""
Base.@kwdef struct ThunkOptions
    single::Int = 0
    proclist = nothing
    procutil::Dict{Type,Any} = Dict{Type,Any}()
    allow_errors::Bool = false
    checkpoint = nothing
    restore = nothing
end

# Eager scheduling
include("eager.jl")

"""
    merge(sopts, topts)

Combine `SchedulerOptions` and `ThunkOptions` into a new `ThunkOptions`.
"""
function merge(sopts::SchedulerOptions, topts::ThunkOptions)
    single = topts.single != 0 ? topts.single : sopts.single
    allow_errors = sopts.allow_errors || topts.allow_errors
    proclist = topts.proclist !== nothing ? topts.proclist : sopts.proclist
    ThunkOptions(single, proclist, topts.procutil, allow_errors, topts.checkpoint, topts.restore)
end
merge(sopts::SchedulerOptions, ::Nothing) =
    ThunkOptions(sopts.single, sopts.proclist, Dict{Type,Any}())

function isrestricted(task::Thunk, proc::OSProc)
    if (task.options !== nothing) && (task.options.single != 0) &&
       (task.options.single != proc.pid)
        return true
    end
    return false
end

function cleanup(ctx)
end

const WORKER_MONITOR_LOCK = Threads.ReentrantLock()
const WORKER_MONITOR_TASKS = Dict{Int,Task}()
const WORKER_MONITOR_CHANS = Dict{Int,Dict{UInt64,RemoteChannel}}()
function init_proc(state, p, log_sink)
    # Initialize pressure and capacity
    gproc = OSProc(p.pid)
    lock(state.lock) do
        state.worker_time_pressure[p.pid] = Dict{Processor,UInt64}()
        for proc in get_processors(gproc)
            state.worker_time_pressure[p.pid][proc] = 0
        end

        state.worker_storage_pressure[p.pid] = Dict{StorageResource,UInt64}()
        state.worker_storage_capacity[p.pid] = Dict{StorageResource,UInt64}()
        for storage in get_storage_resources(gproc)
            pressure, capacity = remotecall_fetch(gproc.pid, storage) do storage
                storage_pressure(storage), storage_capacity(storage)
            end
            state.worker_storage_pressure[p.pid][storage] = pressure
            state.worker_storage_capacity[p.pid][storage] = capacity
        end

        state.worker_loadavg[p.pid] = (0.0, 0.0, 0.0)
    end
    #= FIXME: Is it safe/correct to defer this?
    cap = remotecall(capacity, p.pid)
    @async begin
        cap = fetch(cap) * UInt64(1e9)
        lock(state.lock) do
            state.worker_capacity[p.pid] = cap
        end
    end
    =#
    lock(WORKER_MONITOR_LOCK) do
        wid = p.pid
        if !haskey(WORKER_MONITOR_TASKS, wid)
            t = @async begin
                try
                    # Wait until this connection is terminated
                    remotecall_fetch(sleep, wid, typemax(UInt64))
                catch err
                    if err isa ProcessExitedException
                        lock(WORKER_MONITOR_LOCK) do
                            d = WORKER_MONITOR_CHANS[wid]
                            for uid in keys(d)
                                put!(d[uid], (wid, OSProc(wid), nothing, (ProcessExitedException(wid), nothing)))
                            end
                            empty!(d)
                            delete!(WORKER_MONITOR_CHANS, wid)
                        end
                    end
                end
            end
            WORKER_MONITOR_TASKS[wid] = t
            WORKER_MONITOR_CHANS[wid] = Dict{UInt64,RemoteChannel}()
        end
        WORKER_MONITOR_CHANS[wid][state.uid] = state.chan
    end

    # Setup worker-to-scheduler channels
    inp_chan = RemoteChannel(p.pid)
    out_chan = RemoteChannel(p.pid)
    lock(state.lock) do
        state.worker_chans[p.pid] = (inp_chan, out_chan)
    end
end
function _cleanup_proc(uid, log_sink)
end
function cleanup_proc(state, p, log_sink)
    lock(WORKER_MONITOR_LOCK) do
        wid = p.pid
        if haskey(WORKER_MONITOR_CHANS, wid)
            delete!(WORKER_MONITOR_CHANS[wid], state.uid)
            remote_do(_cleanup_proc, wid, state.uid, log_sink)
        end
    end
end

"Process-local condition variable (and lock) indicating task completion."
const TASK_SYNC = Threads.Condition()

"Process-local set of running task IDs."
const TASKS_RUNNING = Set{Int}()

"Process-local dictionary tracking per-processor total time utilization."
const PROCESSOR_TIME_UTILIZATION = Dict{UInt64,Dict{Processor,Ref{UInt64}}}()

# TODO: "Process-local count of actively-executing Dagger tasks per processor type."

"""
    MaxUtilization

Indicates a thunk that uses all processors of a given type.
"""
struct MaxUtilization end

function compute_dag(ctx, d::Thunk; options=SchedulerOptions())
    if options === nothing
        options = SchedulerOptions()
    end
    ctx.options = options
    if options.restore !== nothing
        try
            result = options.restore()
            if result isa Chunk
                return result
            elseif result !== nothing
                throw(ArgumentError("Invalid restore return type: $(typeof(result))"))
            end
        catch err
            report_catch_error(err, "Scheduler restore failed")
        end
    end
    master = OSProc(myid())
    timespan_start(ctx, :scheduler_init, 0, master)

    chan = RemoteChannel(()->Channel(1024))
    deps = dependents(d)
    ord = order(d, noffspring(deps))

    node_order = x -> -get(ord, x, 0)
    state = start_state(deps, node_order, chan)

    # setup thunk_dict mappings
    for node in filter(istask, keys(deps))
        state.thunk_dict[node.id] = WeakThunk(node)
        for dep in deps[node]
            state.thunk_dict[dep.id] = WeakThunk(dep)
        end
    end

    # Initialize workers
    @sync for p in procs_to_use(ctx)
        @async begin
            try
                init_proc(state, p, ctx.log_sink)
            catch err
                @error "Error initializing worker $p" exception=(err,catch_backtrace())
                remove_dead_proc!(ctx, state, p)
            end
        end
    end

    # setup dynamic listeners
    dynamic_listener!(ctx, state)

    timespan_finish(ctx, :scheduler_init, 0, master)

    # start off some tasks
    # Note: procs_state may be different things for different contexts. Don't touch it out here!
    procs_state = assign_new_procs!(ctx, state, procs_to_use(ctx))

    safepoint(state)

    # Loop while we still have thunks to execute
    while !isempty(state.ready) || !isempty(state.running)
        procs_state = assign_new_procs!(ctx, state, procs_state)
        if !isempty(state.ready)
            # Nothing running, so schedule up to N thunks, 1 per N workers
            schedule!(ctx, state, procs_state)
        end

        # This is a bit redundant as the @async task below does basically the
        # same job Without it though, testing of process modification becomes
        # non-deterministic (due to sleep in CI environment) which is why it is
        # still here.
        procs_state = assign_new_procs!(ctx, state, procs_state)

        check_integrity(ctx)

        # Check periodically for new workers in a parallel task so that we
        # don't accidentally end up having to wait for `take!(chan)` on some
        # large task before new workers are put to work. Locking is used to
        # stop this task as soon as something pops out from the channel to
        # minimize risk that the task schedules thunks simultaneously as the
        # main task (after future refactoring).
        newtasks_lock = ReentrantLock()
        @async while !isempty(state.ready) || !isempty(state.running)
            sleep(1)
            islocked(newtasks_lock) && return
            procs_state = lock(newtasks_lock) do
                assign_new_procs!(ctx, state, procs_state)
            end
        end

        isempty(state.running) && continue
        timespan_start(ctx, :take, 0, 0)
        chan_value = take!(chan) # get result of completed thunk
        timespan_finish(ctx, :take, 0, 0)
        if chan_value isa RescheduleSignal
            continue
        end
        pid, proc, thunk_id, (res, metadata) = chan_value
        gproc = OSProc(pid)
        lock(newtasks_lock) # This waits for any assign_new_procs! above to complete and then shuts down the task
        safepoint(state)
        lock(state.lock) do
            thunk_failed = false
            if res isa Exception
                if unwrap_nested_exception(res) isa ProcessExitedException
                    @warn "Worker $(pid) died, rescheduling work"

                    # Remove dead worker from procs list
                    remove_dead_proc!(ctx, state, gproc)

                    handle_fault(ctx, state, gproc)
                    return # effectively `continue`
                else
                    if ctx.options.allow_errors || unwrap_weak_checked(state.thunk_dict[thunk_id]).options.allow_errors
                        thunk_failed = true
                    else
                        throw(res)
                    end
                end
            end
            node = unwrap_weak_checked(state.thunk_dict[thunk_id])
            if metadata !== nothing
                state.worker_time_pressure[pid][proc] = metadata.time_pressure
                to_storage = metadata.storage
                state.worker_storage_pressure[pid][to_storage] = metadata.storage_pressure
                state.worker_storage_capacity[pid][to_storage] = metadata.storage_capacity
                state.worker_loadavg[pid] = metadata.loadavg
                sig = signature(node, state)
                state.signature_time_cost[sig] = (metadata.threadtime + get(state.signature_time_cost, sig, 0)) ÷ 2
                state.signature_alloc_cost[sig] = (metadata.gc_allocd + get(state.signature_alloc_cost, sig, 0)) ÷ 2
                if metadata.transfer_rate !== nothing
                    state.transfer_rate[] = (state.transfer_rate[] + metadata.transfer_rate) ÷ 2
                end
            end
            state.cache[node] = res
            state.errored[node] = thunk_failed
            if node.options !== nothing && node.options.checkpoint !== nothing
                try
                    node.options.checkpoint(node, res)
                catch err
                    report_catch_error(err, "Thunk checkpoint failed")
                end
            end

            timespan_start(ctx, :finish, thunk_id, (;thunk_id))
            finish_task!(ctx, state, node, thunk_failed)
            timespan_finish(ctx, :finish, thunk_id, (;thunk_id))
        end

        safepoint(state)
    end
    @assert !isready(state.chan)
    close(state.chan)
    notify(state.halt)
    @sync for p in procs_to_use(ctx)
        @async cleanup_proc(state, p, ctx.log_sink)
    end
    value = state.cache[d] # TODO: move(OSProc(), state.cache[d])
    if get(state.errored, d, false)
        throw(value)
    end
    if options.checkpoint !== nothing
        try
            options.checkpoint(value)
        catch err
            report_catch_error(err, "Scheduler checkpoint failed")
        end
    end
    value
end

function procs_to_use(ctx, options=ctx.options)
    return if options.single !== 0
        @assert options.single in vcat(1, workers()) "Sch option `single` must specify an active worker ID."
        OSProc[OSProc(options.single)]
    else
        procs(ctx)
    end
end

check_integrity(ctx) = @assert !isempty(procs_to_use(ctx)) "No suitable workers available in context."

struct SchedulingException <: Exception
    reason::String
end
function Base.show(io::IO, se::SchedulingException)
    print(io, "SchedulingException ($(se.reason))")
end

struct ChunkRedirector
    chunks::Set{Chunk}
    lock::Threads.ReentrantLock
end
ChunkRedirector() = ChunkRedirector(Set{Chunk}(), Threads.ReentrantLock())
Base.push!(cr::ChunkRedirector, c::Chunk) = lock(cr.lock) do
    push!(cr.chunks, c)
end
Base.pop!(cr::ChunkRedirector, c::Chunk) = lock(cr.lock) do
    pop!(cr.chunks, c)
end
Base.collect(cr::ChunkRedirector) = lock(cr.lock) do
    collect(cr.chunks)
end
function Dagger.move(from_proc::Processor, to_proc::Processor, x::Chunk{ChunkRedirector})
    from_wid = Dagger.root_worker_id(from_proc)
    @assert Dagger.root_worker_id(to_proc) == myid()

    # Ask local storage manager for chunk
    real_x = storage_move(from_proc, to_proc, x)

    if real_x === nothing
        # Ask remote storage manager for chunk
        real_x = remotecall(from_wid) do
            storage_move(from_proc, to_proc, x)
        end
        @assert real_x !== nothing "Chunk not found: $x"
    end

    return move(from_proc, to_proc, real_x)
end
function storage_move(from_proc, to_proc, x::Chunk{ChunkRedirector})
    state = STORAGE_STATE
    lock(state.lock) do
        if !haskey(state.cache, x)
            # Unknown chunk
            return nothing
        end
        caches = state.cache[x]
        real_x = nothing

        # Find chunk in CPURAMDevice, and return if found
        for storage in keys(caches)
            if storage isa CPURAMDevice
                real_x = caches[storage]
                break
            end
        end

        if real_x === nothing
            # Find chunk in other storage, and return first found
            chunks = values(caches[storage])
            if length(chunks) > 0
                real_x = first(chunks)
            else
                # No associated chunks
                return nothing
            end
        end

        from_store = real_x.processor
        to_store = best_storage(to_proc)
        return move(from_store, to_store, real_x)
    end
end

best_storage(p) = CPURAMDevice(Dagger.root_worker_id(p))

struct StorageState
    # original to storage to clones
    cache::WeakKeyDict{Chunk,Dict{Processor,Chunk}}

    lock::Threads.ReentrantLock
end

const STORAGE_STATE = StorageState(
    WeakKeyDict{Chunk,Dict{Processor,Chunk}}(),
    Threads.ReentrantLock(),
)

function schedule!(ctx, state, procs=procs_to_use(ctx))
    lock(state.lock) do
        safepoint(state)
        @assert length(procs) > 0

        # Populate the cache if empty
        if state.procs_cache_list[] === nothing
            current = nothing
            for p in map(x->x.pid, procs)
                for proc in get_processors(OSProc(p))
                    next = ProcessorCacheEntry(OSProc(p), proc)
                    if current === nothing
                        current = next
                        current.next = current
                        state.procs_cache_list[] = current
                    else
                        current.next = next
                        current = next
                        current.next = state.procs_cache_list[]
                    end
                end
            end
        end

        function can_use_proc(task, gproc, proc, opts, scope)
            # Check against proclist
            if opts.proclist === nothing
                if !default_enabled(proc)
                    return false
                end
            elseif opts.proclist isa Function
                if !Base.invokelatest(opts.proclist, proc)
                    return false
                end
            elseif opts.proclist isa Vector
                if !(typeof(proc) in opts.proclist)
                    return false
                end
            else
                throw(SchedulingException("proclist must be a Function, Vector, or nothing"))
            end

            # Check against single
            if opts.single != 0
                if gproc.pid != opts.single
                    return false
                end
            end

            # Check scope
            if constrain(scope, Dagger.ExactScope(proc)) isa Dagger.InvalidScope
                return false
            end

            return true
        end
        function has_capacity(p, gp, procutil, sig)
            T = typeof(p)
            # FIXME: MaxUtilization
            est_time_util = round(UInt64, get(procutil, T, 1) * 1e9)
            if (T === Dagger.ThreadProc) && haskey(state.signature_time_cost, sig)
                # Assume that the extra pressure is between estimated and measured
                # TODO: Generalize this to arbitrary processor types
                est_time_util = min(est_time_util, state.signature_time_cost[sig])
            end
            storage = storage_resource(p)
            real_alloc_util = state.worker_storage_pressure[gp][storage]
            real_alloc_cap = state.worker_storage_capacity[gp][storage]
            est_alloc_util = get(state.signature_alloc_cost, sig, 0)
            if est_alloc_util + real_alloc_util > real_alloc_cap
                # TODO: Estimate if cached data can be swapped to storage
                return false, est_time_util, est_alloc_util
            end
            return true, est_time_util, est_alloc_util
        end
        "Like `sum`, but replaces `nothing` entries with the average of non-`nothing` entries."
        function impute_sum(xs)
            all(x->!isa(x, Chunk), xs) && return 0
            avg = round(UInt64, mean(filter(x->x isa Chunk, xs)))
            total = 0
            for x in xs
                total += x !== nothing ? x : avg
            end
            total
        end

        # Schedule tasks
        to_fire = Dict{Tuple{OSProc,<:Processor},Vector{Tuple{Thunk,<:Any,<:Any}}}()
        failed_scheduling = Thunk[]
        tx_rate = state.transfer_rate[]

        # Select a new task and get its options
        task = nothing
        @label pop_task
        if task !== nothing
            timespan_finish(ctx, :schedule, task.id, (;thunk_id=task.id))
        end
        if isempty(state.ready)
            @goto fire_tasks
        end
        task = pop!(state.ready)
        timespan_start(ctx, :schedule, task.id, (;thunk_id=task.id))
        @assert !haskey(state.cache, task)
        opts = merge(ctx.options, task.options)
        sig = signature(task, state)

        # Calculate scope
        scope = if task.f isa Chunk
            task.f.scope
        else
            AnyScope()
        end
        for input in unwrap_weak_checked.(task.inputs)
            chunk = if istask(input)
                state.cache[input]
            elseif input isa Chunk
                input
            else
                nothing
            end
            chunk isa Chunk || continue
            scope = constrain(scope, chunk.scope)
            if scope isa Dagger.InvalidScope
                ex = SchedulingException("Scopes are not compatible: $(scope.x), $(scope.y)")
                state.cache[task] = ex
                state.errored[task] = true
                set_failed!(state, task)
                @goto pop_task
            end
        end

        fallback_threshold = 1024 # TODO: Parameterize this threshold
        if length(procs) > fallback_threshold
            @goto fallback
        end
        # Select processor with highest data locality, if possible
        # TODO: Account for process-local data movement
        inputs = filter(t->istask(t)||isa(t,Chunk), unwrap_weak_checked.(task.inputs))
        chunks = [istask(input) ? state.cache[input] : input for input in inputs]
        #local_procs = unique(map(c->c isa Chunk ? processor(c) : OSProc(), chunks))
        local_procs = vcat([collect(Dagger.get_processors(gp)) for gp in procs]...)
        if length(local_procs) > fallback_threshold
            @goto fallback
        end
        affinities = Dict(proc=>impute_sum([affinity(chunk)[2] for chunk in filter(c->isa(c,Chunk)&&get_parent(processor(c))==get_parent(proc), chunks)]) for proc in local_procs)
        # Estimate cost to move data and get scheduled
        costs = Dict(proc=>state.worker_time_pressure[get_parent(proc).pid][proc]+(aff/tx_rate) for (proc,aff) in affinities)
        sort!(local_procs, by=p->costs[p])
        scheduled = false
        for proc in local_procs
            gproc = get_parent(proc)
            if can_use_proc(task, gproc, proc, opts, scope)
                has_cap, est_time_util, est_alloc_util = has_capacity(proc, gproc.pid, opts.procutil, sig)
                if has_cap
                    # Schedule task onto proc
                    # FIXME: est_time_util = est_time_util isa MaxUtilization ? cap : est_time_util
                    push!(get!(()->Vector{Tuple{Thunk,<:Any,<:Any}}(), to_fire, (gproc, proc)), (task, est_time_util, est_alloc_util))
                    state.worker_time_pressure[gproc.pid][proc] += est_time_util
                    @goto pop_task
                end
            end
        end
        state.cache[task] = SchedulingException("No processors available, try making proclist more liberal")
        state.errored[task] = true
        set_failed!(state, task)
        @goto pop_task

        # Fast fallback algorithm, used when the smarter cost model algorithm
        # would be too expensive
        @label fallback
        selected_entry = nothing
        entry = state.procs_cache_list[]
        cap, extra_util = nothing, nothing
        procs_found = false
        # N.B. if we only have one processor, we need to select it now
        if can_use_proc(task, entry.gproc, entry.proc, opts, scope)
            has_cap, est_time_util, est_alloc_util = has_capacity(entry.proc, entry.gproc.pid, opts.procutil, sig)
            if has_cap
                selected_entry = entry
            else
                procs_found = true
                entry = entry.next
            end
        else
            entry = entry.next
        end
        while selected_entry === nothing
            if entry === state.procs_cache_list[]
                # Exhausted all procs
                if procs_found
                    push!(failed_scheduling, task)
                else
                    state.cache[task] = SchedulingException("No processors available, try making proclist more liberal")
                    state.errored[task] = true
                    set_failed!(state, task)
                end
                @goto pop_task
            end

            if can_use_proc(task, entry.gproc, entry.proc, opts, scope)
                has_cap, est_time_util, est_alloc_util = has_capacity(entry.proc, entry.gproc.pid, opts.procutil, sig)
                if has_cap
                    # Select this processor
                    selected_entry = entry
                else
                    # We could have selected it otherwise
                    procs_found = true
                    entry = entry.next
                end
            else
                # Try next processor
                entry = entry.next
            end
        end
        @assert selected_entry !== nothing

        # Schedule task onto proc
        gproc, proc = entry.gproc, entry.proc
        est_time_util = est_time_util isa MaxUtilization ? cap : est_time_util
        push!(get!(()->Vector{Tuple{Thunk,<:Any,<:Any}}(), to_fire, (gproc, proc)), (task, est_time_util, est_alloc_util))

        # Proceed to next entry to spread work
        if !ctx.options.round_robin
            @warn "Round-robin mode is always on"
        end
        state.procs_cache_list[] = state.procs_cache_list[].next
        @goto pop_task

        # Fire all newly-scheduled tasks
        @label fire_tasks
        @sync for gpp in keys(to_fire)
            @async fire_tasks!(ctx, to_fire[gpp], gpp, state)
        end
        append!(state.ready, failed_scheduling)
    end
end

# Main responsibility of this function is to check if new procs have been pushed to the context
function assign_new_procs!(ctx, state, assignedprocs)
    ps = procs_to_use(ctx)
    # Must track individual procs to handle the case when procs are removed
    diffps = setdiff(ps, assignedprocs)
    if !isempty(diffps)
        for p in diffps
            init_proc(state, p, ctx.log_sink)
        end
        state.procs_cache_list[] = nothing
        schedule!(ctx, state, diffps)
    end
    return ps
end

shall_remove_proc(ctx, proc) = proc ∉ procs_to_use(ctx)

function remove_dead_proc!(ctx, state, proc, options=ctx.options)
    @assert options.single !== proc.pid "Single worker failed, cannot continue."
    rmprocs!(ctx, [proc])
    delete!(state.worker_time_pressure, proc.pid)
    delete!(state.worker_storage_pressure, proc.pid)
    delete!(state.worker_storage_capacity, proc.pid)
    delete!(state.worker_loadavg, proc.pid)
    delete!(state.worker_chans, proc.pid)
    state.procs_cache_list[] = nothing
end

function pop_with_affinity!(ctx, tasks, proc)
    # TODO: use the size
    parent_affinity_procs = Vector(undef, length(tasks))
    # parent_affinity_sizes = Vector(undef, length(tasks))
    for i=length(tasks):-1:1
        t = tasks[i]
        aff = affinity(t)
        aff_procs = first.(aff)
        if proc in aff_procs
            if !isrestricted(t,proc)
                deleteat!(tasks, i)
                return t
            end
        end
        parent_affinity_procs[i] = aff_procs
    end
    for i=length(tasks):-1:1
        # use up tasks without affinities
        # let the procs with the respective affinities pick up
        # other tasks
        aff_procs = parent_affinity_procs[i]
        if isempty(aff_procs)
            t = tasks[i]
            if !isrestricted(t,proc)
                deleteat!(tasks, i)
                return t
            end
        end
        if all(!(p in aff_procs) for p in procs(ctx))
            # no proc is ever going to ask for it
            t = tasks[i]
            if !isrestricted(t,proc)
                deleteat!(tasks, i)
                return t
            end
        end
    end
    return nothing
end

function finish_task!(ctx, state, node, thunk_failed)
    pop!(state.running, node)
    delete!(state.running_on, node)
    if thunk_failed
        set_failed!(state, node)
    end
    if node.cache
        node.cache_ref = state.cache[node]
    end
    schedule_dependents!(state, node, thunk_failed)
    fill_registered_futures!(state, node, thunk_failed)

    to_evict = cleanup_inputs!(state, node)
    if node.f isa Chunk
        # FIXME: Check the graph for matching chunks
        push!(to_evict, node.f)
    end
    if haskey(state.waiting_data, node) && isempty(state.waiting_data[node])
        delete!(state.waiting_data, node)
    end
    evict_all_chunks!(ctx, to_evict)
end

function evict_all_chunks!(ctx, to_evict)
    if !isempty(to_evict)
        @sync for w in map(p->p.pid, procs_to_use(ctx))
            @async remote_do(evict_chunks!, w, ctx.log_sink, to_evict)
        end
    end
end
function evict_chunks!(log_sink, chunks::Set{Chunk})
    #= FIXME: Tell storage scheduler it can do (lazy) cleanup?
    ctx = Context(;log_sink)
    sstate = STORAGE_STATE
    for chunk in chunks
        timespan_start(ctx, :evict, myid(), (;data=chunk))
        haskey(CHUNK_CACHE, chunk) && delete!(CHUNK_CACHE, chunk)
        timespan_finish(ctx, :evict, myid(), (;data=chunk))
    end
    =#
    nothing
end

fire_task!(ctx, thunk::Thunk, p, state; time_util=10^9, alloc_util=10^6) =
    fire_task!(ctx, (thunk, time_util, alloc_util), p, state)
fire_task!(ctx, (thunk, time_util, alloc_util)::Tuple{Thunk,<:Any}, p, state) =
    fire_tasks!(ctx, [(thunk, time_util, alloc_util)], p, state)
function fire_tasks!(ctx, thunks::Vector{<:Tuple}, (gproc, proc), state)
    to_send = []
    for (thunk, time_util, alloc_util) in thunks
        push!(state.running, thunk)
        state.running_on[thunk] = gproc
        if thunk.cache && thunk.cache_ref !== nothing
            # the result might be already cached
            data = unrelease(thunk.cache_ref) # ask worker to keep the data around
                                              # till this compute cycle frees it
            if data !== nothing
                # cache hit
                state.cache[thunk] = data
                thunk_failed = get(state.errored, thunk, false)
                finish_task!(ctx, state, thunk, thunk_failed)
                continue
            else
                # cache miss
                thunk.cache_ref = nothing
            end
        end
        if thunk.options !== nothing && thunk.options.restore !== nothing
            try
                result = thunk.options.restore(thunk)
                if result isa Chunk
                    state.cache[thunk] = result
                    state.errored[thunk] = false
                    finish_task!(ctx, state, thunk, false)
                    continue
                elseif result !== nothing
                    throw(ArgumentError("Invalid restore return type: $(typeof(result))"))
                end
            catch err
                report_catch_error(err, "Thunk restore failed")
            end
        end

        ids = convert(Vector{Int}, map(enumerate(thunk.inputs)) do (idx,x)
            istask(x) ? unwrap_weak_checked(x).id : -idx
        end)
        pushfirst!(ids, 0)

        data = convert(Vector{Any}, map(Any[thunk.inputs...]) do x
            istask(x) ? state.cache[unwrap_weak_checked(x)] : x
        end)
        pushfirst!(data, thunk.f)
        toptions = thunk.options !== nothing ? thunk.options : ThunkOptions()
        options = merge(ctx.options, toptions)
        @assert (options.single == 0) || (gproc.pid == options.single)
        # TODO: Set `sch_handle.tid.ref` to the right `DRef`
        sch_handle = SchedulerHandle(ThunkID(thunk.id, nothing), state.worker_chans[gproc.pid]...)

        # TODO: De-dup common fields (log_sink, uid, etc.)
        push!(to_send, (thunk.id, time_util, alloc_util, fn_type(thunk.f), data, thunk.get_result,
                        thunk.persist, thunk.cache, thunk.meta, options, ids,
                        (log_sink=ctx.log_sink, profile=ctx.profile),
                        sch_handle, state.uid))
    end
    timespan_start(ctx, :fire_multi, 0, 0)
    try
        remotecall_wait(do_tasks, gproc.pid, proc, state.chan, to_send)
    catch
        # We might get a deserialization error due to something not being
        # defined on the worker; in this case, we re-fire one task at a time to
        # determine which task failed
        for ts in to_send
            thunk_id = ts[1]
            try
                remotecall_wait(do_tasks, gproc.pid, proc, state.chan, [ts])
            catch err
                bt = catch_backtrace()
                put!(state.chan, (gproc.pid, proc, thunk_id, (CapturedException(err, bt), nothing)))
            end
        end
    end
    timespan_finish(ctx, :fire_multi, 0, 0)
end

"""
    do_tasks(to_proc, chan, tasks)

Executes a batch of tasks on `to_proc`.
"""
function do_tasks(to_proc, chan, tasks)
    for task in tasks
        thunk_id = task[1]
        should_launch = lock(TASK_SYNC) do
            # Already running; don't try to re-launch
            if !(thunk_id in TASKS_RUNNING)
                push!(TASKS_RUNNING, thunk_id)
                true
            else
                false
            end
        end
        should_launch || continue
        @async begin
            try
                result = do_task(to_proc, task...)
                put!(chan, (myid(), to_proc, thunk_id, result))
            catch ex
                bt = catch_backtrace()
                put!(chan, (myid(), to_proc, thunk_id, (CapturedException(ex, bt), nothing)))
            end
        end
    end
end
"Executes a single task on `to_proc`."
function do_task(to_proc, thunk_id, est_time_util, est_alloc_util, Tf, data, send_result, persist, cache, meta, options, ids, ctx_vars, sch_handle, uid)
    ctx = Context(Processor[]; log_sink=ctx_vars.log_sink, profile=ctx_vars.profile)

    from_proc = OSProc()
    to_storage = storage_resource(to_proc)
    f = isdefined(Tf, :instance) ? Tf.instance : nothing

    # Wait for required resources to become available
    timespan_start(ctx, :storage_wait, thunk_id, (;f, to_proc, to_storage))
    storage_cap = storage_capacity(to_storage)
    real_time_util = Ref{UInt64}(0)
    real_alloc_util = UInt64(0)
    while true
        lock(TASK_SYNC)

        # Get current time utilization for the selected processor
        time_dict = get!(()->Dict{Processor,Ref{UInt64}}(), PROCESSOR_TIME_UTILIZATION, uid)
        real_time_util = get!(()->Ref{UInt64}(UInt64(0)), time_dict, to_proc)

        # Get current allocation utilization for the selected processor's memory resource
        real_alloc_util = storage_cap - storage_available(to_storage)

        # Check if we'll go over memory capacity from running this thunk, and
        # wait for free storage if necessary
        if est_alloc_util + real_alloc_util > storage_cap
            # Fully utilized, wait and re-check
            @debug "[$(myid())] $f ($thunk_id) Waiting for free $to_storage: $est_alloc_util | $real_alloc_util/$storage_cap"
            wait(TASK_SYNC)
            unlock(TASK_SYNC)
        else
            # Sufficient free storage is available, prepare for execution
            @debug "[$(myid())] $f ($thunk_id) Using available $to_storage: $est_alloc_util | $real_alloc_util/$storage_cap"
            unlock(TASK_SYNC)
            break
        end
    end
    timespan_finish(ctx, :storage_wait, thunk_id, (;f, to_proc, to_storage))

    # Initiate data transfers for function and arguments
    transfer_time = Threads.Atomic{UInt64}(0)
    transfer_size = Threads.Atomic{UInt64}(0)
    fetched = if meta
        data
    else
        scache = STORAGE_STATE.cache
        fetch_report.(map(Iterators.zip(data,ids)) do (x, id)
            @async begin
                timespan_start(ctx, :move, (;thunk_id, id), (;f, id, data=x))
                x = if x isa Chunk
                    if haskey(scache, x)
                        get!(scache[x], to_proc) do
                            # TODO: Choose "closest" processor of same type first
                            some_proc = first(keys(scache[x]))
                            some_x = scache[x][some_proc]
                            move(some_proc, to_proc, some_x)
                        end
                    else
                        time_start = time_ns()
                        _x = move(to_proc, x)
                        time_finish = time_ns()
                        if x.handle.size !== nothing
                            Threads.atomic_add!(transfer_time, time_finish - time_start)
                            Threads.atomic_add!(transfer_size, x.handle.size)
                        end
                        scache[x] = Dict{Processor,Any}()
                        scache[x][to_proc] = _x
                        _x
                    end
                else
                    move(to_proc, x)
                end
                timespan_finish(ctx, :move, (;thunk_id, id), (;f, id, data=x); tasks=[Base.current_task()])
                return x
            end
        end)
    end
    f = popfirst!(fetched)

    #= FIXME: If MaxUtilization, stop processors and wait
    if (est_time_util isa MaxUtilization) && (real_time_util > 0)
        # FIXME: Stop processors
        # FIXME: Wait on processors to stop
        est_time_util = count(c->typeof(c)===typeof(to_proc), children(from_proc))
    end
    =#

    real_time_util[] += est_time_util
    timespan_start(ctx, :compute, thunk_id, (;f, to_proc))
    res = nothing

    # Start counting time and GC allocations
    threadtime_start = cputhreadtime()
    gcnum_start = Base.gc_num()

    result_meta = try
        # Set TLS variables
        Dagger.set_tls!((
            sch_uid=uid,
            sch_handle=sch_handle,
            processor=to_proc,
            time_utilization=est_time_util,
            alloc_utilization=est_alloc_util,
        ))

        # Execute
        res = execute!(to_proc, f, fetched...)

        # Construct result
        if send_result || meta
            res
        else
            # Wrap result in `Chunk`
            cache = persist ? true : cache
            storage = CPURAMDevice(myid())
            real_chunk = tochunk(res, storage; persist, cache)

            # Setup redirection `Chunk` for storage handling
            redir = ChunkRedirector()
            push!(redir, real_chunk)
            redir_chunk = tochunk(redir, to_proc; persist, cache)

            # Register with storage system
            storage_cache = Dict{Processor,Chunk}()
            storage_cache[storage] = real_chunk
            lock(STORAGE_STATE.lock) do
                STORAGE_STATE.cache[redir_chunk] = storage_cache
            end

            redir_chunk
        end
    catch ex
        bt = catch_backtrace()
        RemoteException(myid(), CapturedException(ex, bt))
    end
    threadtime = cputhreadtime() - threadtime_start
    # FIXME: This is not a realistic measure of max. required memory
    gc_allocd = max(UInt64(Base.gc_num().allocd) - UInt64(gcnum_start.allocd), UInt64(0))
    timespan_finish(ctx, :compute, thunk_id, (;f, to_proc); tasks=Dagger.prof_tasks_take!(thunk_id))
    lock(TASK_SYNC) do
        real_time_util[] -= est_time_util
        pop!(TASKS_RUNNING, thunk_id)
        notify(TASK_SYNC)
    end
    @debug "[$(myid())] $f ($thunk_id) Releasing $to_storage: $est_alloc_util | $real_alloc_util/$storage_cap"
    metadata = (
        time_pressure=real_time_util[],
        storage=to_storage,
        storage_pressure=real_alloc_util,
        storage_capacity=storage_cap,
        loadavg=((Sys.loadavg()...,) ./ Sys.CPU_THREADS),
        threadtime=threadtime,
        gc_allocd=gc_allocd,
        transfer_rate=transfer_time[] > 0 ? round(UInt64, transfer_size[] / (transfer_time[] / 10^9)) : nothing,
    )
    (result_meta, metadata)
end

end # module Sch
