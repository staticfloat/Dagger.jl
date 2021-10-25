using Serialization

export domain, UnitDomain, project, alignfirst, ArrayDomain

import Base: isempty, getindex, intersect, ==, size, length, ndims

"""
    domain(x::T)

Returns metadata about `x`. This metadata will be in the `domain`
field of a Chunk object when an object of type `T` is created as
the result of evaluating a Thunk.
"""
function domain end

"""
    UnitDomain

Default domain -- has no information about the value
"""
struct UnitDomain end

"""
If no `domain` method is defined on an object, then
we use the `UnitDomain` on it. A `UnitDomain` is indivisible.
"""
domain(x::Any) = UnitDomain()

###### Chunk ######

"""
    Chunk

A reference to a piece of data located on a remote worker. `Chunk`s are
typically created with `Dagger.tochunk(data)`, and the data can then be
accessed from any worker with `collect(::Chunk)`. `Chunk`s are
serialization-safe, and use distributed refcounting (provided by
`MemPool.DRef`) to ensure that the data referenced by a `Chunk` won't be GC'd,
as long as a reference exists on some worker.

Each `Chunk` is associated with a given `Dagger.Processor`, which is (in a
sense) the processor that "owns" or contains the data. Calling
`collect(::Chunk)` will perform data movement and conversions defined by that
processor to safely serialize the data to the calling worker.

## Constructors
See [`tochunk`](@ref).
"""
mutable struct Chunk{T, H, P<:Processor, S<:AbstractScope}
    chunktype::Type{T}
    domain
    handle::H
    processor::P
    scope::S
    persist::Bool
end

domain(c::Chunk) = c.domain
chunktype(c::Chunk) = c.chunktype
persist!(t::Chunk) = (t.persist=true; t)
shouldpersist(p::Chunk) = t.persist
processor(c::Chunk) = c.processor
affinity(c::Chunk) = affinity(c.handle)

function unrelease(c::Chunk{<:Any,DRef})
    # set spilltodisk = true if data is still around
    try
        destroyonevict(c.handle, false)
        return c
    catch err
        if isa(err, KeyError)
            return nothing
        else
            rethrow(err)
        end
    end
end
unrelease(c::Chunk) = c

Base.:(==)(c1::Chunk, c2::Chunk) = c1.handle == c2.handle
Base.hash(c::Chunk, x::UInt64) = hash(c.handle, x)

collect_remote(chunk::Chunk) =
    move(chunk.processor, OSProc(), poolget(chunk.handle))

function collect(ctx::Context, chunk::Chunk; options=nothing)
    # delegate fetching to handle by default.
    if chunk.handle isa DRef && !(chunk.processor isa OSProc)
        return remotecall_fetch(collect_remote, chunk.handle.owner, chunk)
    elseif chunk.handle isa FileRef
        return poolget(chunk.handle)
    else
        return move(chunk.processor, OSProc(), chunk.handle)
    end
end
collect(ctx::Context, ref::DRef; options=nothing) =
    move(OSProc(ref.owner), OSProc(), ref)
collect(ctx::Context, ref::FileRef; options=nothing) =
    poolget(ref) # FIXME: Do move call

# Unwrap Chunk, DRef, and FileRef by default
move(from_proc::Processor, to_proc::Processor, x::Chunk) =
    move(from_proc, to_proc, x.handle)

move(from_proc::Processor, to_proc::Processor, x::Union{DRef,FileRef}) =
    move(from_proc, to_proc, poolget(x))

# Determine from_proc when unspecified
move(to_proc::Processor, chunk::Chunk) =
    move(chunk.processor, to_proc, chunk)
move(to_proc::Processor, d::DRef) =
    move(OSProc(d.owner), to_proc, d)
move(to_proc::Processor, x) =
    move(OSProc(), to_proc, x)

### ChunkIO
affinity(r::DRef) = OSProc(r.owner)=>r.size
function affinity(r::FileRef)
    if haskey(MemPool.who_has_read, r.file)
        Pair{OSProc, UInt64}[OSProc(dref.owner) => r.size for dref in MemPool.who_has_read[r.file]]
    elseif r.force_pid[] !== nothing
        Pair{OSProc, UInt64}[OSProc(r.force_pid[]) => 1]
    else
        Pair{OSProc, UInt64}[OSProc(w) => r.size for w in MemPool.get_workers_at(r.host)]
    end
end

"""
    tochunk(x, proc; persist=false, cache=false) -> Chunk

Create a chunk from sequential object `x` which resides on `proc`.
"""
function tochunk(x::X, proc::P=OSProc(), scope::S=AnyScope(); persist=false, cache=false) where {X,P,S}
    ref = poolset(x, destroyonevict=persist ? false : cache)
    Chunk{X,typeof(ref),P,S}(X, domain(x), ref, proc, scope, persist)
end
tochunk(x::Union{Chunk, Thunk}, proc=nothing, scope=nothing) = x

# Check to see if the node is set to persist
# if it is foce can override it
function free!(s::Chunk{X,DRef,P,S}; force=true, cache=false) where {X,P,S}
    if force || !s.persist
        if cache
            try
                destroyonevict(s.handle, true) # keep around, but remove when evicted
            catch err
                isa(err, KeyError) || rethrow(err)
            end
        end
    end
end
free!(x; force=true,cache=false) = x # catch-all for non-chunks

function savechunk(data, dir, f)
    sz = open(joinpath(dir, f), "w") do io
        serialize(io, MemPool.MMWrap(data))
        return position(io)
    end
    fr = FileRef(f, sz)
    proc = OSProc()
    scope = AnyScope() # FIXME: Scoped to this node
    Chunk{typeof(data),typeof(fr),typeof(proc),typeof(scope)}(typeof(data), domain(data), fr, proc, scope, true)
end


Base.@deprecate_binding AbstractPart Union{Chunk, Thunk}
Base.@deprecate_binding Part Chunk
Base.@deprecate parts(args...) chunks(args...)
Base.@deprecate part(args...) tochunk(args...)
Base.@deprecate parttype(args...) chunktype(args...)
