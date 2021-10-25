"""
    StorageResource

The supertype for all storage resources. Any subtype represents a physical or
logical resource (RAM, disk, NAS, compressed RAM, etc.) where data can be
stored, and which has some current usage level (measured in bytes) and a
maximum capacity.

`Processor` subtypes may have an associated `StorageResource` object (or objects)
which are the canonical storage locations for data affiliated with the
processor.

`StorageResource`s can not be directly manipulated; a `StorageDevice` is
required to interface with the resource.
"""
abstract type StorageResource end
storage_available(::StorageResource) = 0
storage_pressure(s::StorageResource) = storage_capacity(s) - storage_available(s)
storage_capacity(::StorageResource) = 0

"""
    StorageDevice <: Processor

The supertype for all storage devices. Any subtype represents some method of
utilizing a storage device, and has an associated `StorageResource`
representing the resource being utilized.
"""
abstract type StorageDevice <: Processor end
get_parent(s::StorageDevice) = s

const STORAGE_RESOURCE_CALLBACKS = Dict{Symbol,Any}()
const STORAGE_DEVICE_CALLBACKS = Dict{Symbol,Any}()
const OSPROC_STORAGE_RESOURCE_CACHE = Dict{Int,Set{StorageResource}}()
const OSPROC_STORAGE_DEVICE_CACHE = Dict{Int,Set{StorageDevice}}()

add_storage_resource_callback!(func, name::String) =
    add_storage_resource_callback!(func, Symbol(name))
function add_storage_resource_callback!(func, name::Symbol)
    Dagger.STORAGE_RESOURCE_CALLBACKS[name] = func
    delete!(OSPROC_STORAGE_RESOURCE_CACHE, myid())
end
add_storage_device_callback!(func, name::String) =
    add_storage_device_callback!(func, Symbol(name))
function add_storage_device_callback!(func, name::Symbol)
    Dagger.STORAGE_DEVICE_CALLBACKS[name] = func
    delete!(OSPROC_STORAGE_DEVICE_CACHE, myid())
end

delete_storage_resource_callback!(name::String) =
    delete_storage_resource_callback!(Symbol(name))
function delete_storage_resource_callback!(name::Symbol)
    delete!(Dagger.STORAGE_RESOURCE_CALLBACKS, name)
    delete!(OSPROC_STORAGE_RESOURCE_CACHE, myid())
end
delete_storage_device_callback!(name::String) =
    delete_storage_device_callback!(Symbol(name))
function delete_storage_device_callback!(name::Symbol)
    delete!(Dagger.STORAGE_DEVICE_CALLBACKS, name)
    delete!(OSPROC_STORAGE_DEVICE_CACHE, myid())
end

"""
    CPURAMStorage <: StorageResource

Represents CPU RAM (usually DRAM).
"""
struct CPURAMStorage <: StorageResource
    owner::Int
end
storage_available(::CPURAMStorage) = Sys.free_memory()
storage_capacity(::CPURAMStorage) = Sys.total_memory()

"""
    CPURAMDevice <: StorageDevice

Stores data plainly in CPU RAM, according to Julia's memory layout.
"""
struct CPURAMDevice <: StorageDevice
    owner::Int
end
get_parent(s::CPURAMDevice) = OSProc(s.owner)
storage_resource(s::CPURAMDevice) = CPURAMStorage(s.owner)
function move(from::CPURAMDevice, to::CPURAMDevice, x::DRef)
    @assert to.owner == myid()
    poolget(x)
end

"""
    FilesystemStorage <: StorageResource

Represents a filesystem mounted at a given path.
"""
struct FilesystemStorage <: StorageResource
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
    return vfs.f_bavail * vfs.f_bsize
end
function storage_capacity(s::FilesystemStorage)
    vfs = statvfs(s.mountpoint)
    return vfs.f_blocks * vfs.f_bsize
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
    SerializedFilesystemDevice <: StorageDevice

Represents data stored on a `StorageFilesystem` in a given directory.
"""
struct SerializedFilesystemDevice <: StorageDevice
    filesystem::FilesystemStorage
    path::String
    id_ctr::Threads.Atomic{Int}
    function SerializedFilesystemDevice(filesystem::FilesystemStorage, path::String)
        @assert isdir(path) "Not a valid directory: $path"
        new(filesystem, path, Threads.Atomic{Int}(1))
    end
end
# FIXME: get_parent
storage_resource(s::SerializedFilesystemDevice) = s.filesystem
function move(from::CPURAMDevice, to::SerializedFilesystemDevice, x::DRef)
    @assert from.owner == to.owner == myid()
    id = Threads.atomic_add!(to.id_ctr, 1)
    path = joinpath(to.path, "$id")
    serialize(path, poolget(x))
    Dagger.tochunk(ExpiringFile(path), to)
end
function move(from::SerializedFilesystemDevice, to::CPURAMDevice, x::DRef)
    @assert from.owner == to.owner == myid()
    path = poolget(x)
    deserialize(path.path)
end
