"""
    Storage <: Processor

The supertype for all storage objects. Any subtype represents some storage
device (RAM, disk, NAS, compressed RAM, etc.) where data can be stored, which
has some maximum capacity.

`Processor` subtypes may have an associated `Storage` object (or objects) which
are the canonical storage locations for data affiliated with the processor.
"""
abstract type Storage <: Processor end
get_parent(s::Storage) = s
storage_available(::Storage) = 0
storage_capacity(::Storage) = 0

const STORAGE_CALLBACKS = Dict{Symbol,Any}()
const OSPROC_STORAGE_CACHE = Dict{Int,Vector{Storage}}()

add_storage_callback!(func, name::String) =
    add_storage_callback!(func, Symbol(name))
function add_storage_callback!(func, name::Symbol)
    Dagger.STORAGE_CALLBACKS[name] = func
    delete!(OSPROC_STORAGE_CACHE, myid())
end
delete_storage_callback!(name::String) =
    delete_storage_callback!(Symbol(name))
function delete_storage_callback!(name::Symbol)
    delete!(Dagger.STORAGE_CALLBACKS, name)
    delete!(OSPROC_STORAGE_CACHE, myid())
end

"""
    CPURAMStorage <: Storage

Represents data stored plainly in CPU RAM (usually DRAM).
"""
struct CPURAMStorage <: Storage
    owner::Int
end
get_parent(s::CPURAMStorage) = OSProc(s.owner)
storage_available(::CPURAMStorage) = Sys.free_memory()
storage_capacity(::CPURAMStorage) = Sys.total_memory()
function move(from::CPURAMStorage, to::CPURAMStorage, x::DRef)
    @assert to.owner == myid()
    poolget(x)
end

"""
    FilesystemStorage <: Storage

Represents data stored on a filesystem at a given mountpoint. Does not have a
canonical data representation.
"""
struct FilesystemStorage <: Storage
    owner::Int
    mountpoint::String
    function FilesystemStorage(mountpoint::String)
        @assert ismount(mountpoint) "Not a mountpoint: $mountpoint"
        new(myid(), mountpoint)
    end
end
get_parent(s::FilesystemStorage) = OSProc(s.owner)
function storage_available(s::FilesystemStorage)
    vfs = statvfs(s.mountpoint)
    return vfs.f_frsize * vfs.f_blocks
end
function storage_capacity(s::FilesystemStorage)
    vfs = statvfs(s.mountpoint)
    return vfs.f_frsize * vfs.f_blocks
end

"An automatically-deleting file."
mutable struct ExpiringFile
    path::String
    function ExpiringFile(path)
        ef = new(path)
        finalizer(ef) do ef
            try
                rm(path)
            catch
                @warn "Failed to remove $path"
            end
        end
        ef
    end
end

"""
    SerializedFilesystemStorage <: FilesystemStorage

Represents data stored on a filesystem in a given directory.
"""
struct SerializedFilesystemStorage <: Storage
    parent::FilesystemStorage
    path::String
    id_ctr::Threads.Atomic{Int}
    function SerializedFilesystemStorage(parent::FilesystemStorage, path::String)
        @assert isdir(path) "Not a valid directory: $path"
        new(parent, path, Threads.Atomic{Int}(1))
    end
end
get_parent(s::SerializedFilesystemStorage) = s.parent
storage_available(s::SerializedFilesystemStorage) = storage_available(s.parent)
storage_capacity(s::SerializedFilesystemStorage) = storage_capacity(s.parent)
function move(from::CPURAMStorage, to::SerializedFilesystemStorage, x::DRef)
    @assert from.owner == to.owner == myid()
    id = Threads.atomic_add!(to.id_ctr, 1)
    path = joinpath(to.path, "$id")
    serialize(path, poolget(x))
    Dagger.tochunk(ExpiringFile(path), to)
end
function move(from::SerializedFilesystemStorage, to::CPURAMStorage, x::DRef)
    @assert from.owner == to.owner == myid()
    path = poolget(x)
    deserialize(path.path)
end
