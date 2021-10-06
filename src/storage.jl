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

"""
    FilesystemStorage <: Storage

Represents data stored on a filesystem at a given mountpoint. Does not have a
canonical data representation.
"""
struct FilesystemStorage <: Storage
    owner::Int
    mountpoint::String
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

"""
    SerializationDiskStorage <: FilesystemStorage

Represents data stored on a filesystem in a given directory.
"""
struct SerializationDiskStorage <: Storage
    parent::FilesystemStorage
    path::String
    id_ctr::Threads.Atomic{Int}
    function SerializationDiskStorage(path::String)
        # FIXME: Get parent
        new(parent, path, Threads.Atomic{Int}(1))
    end
end
get_parent(s::SerializationDiskStorage) = s.parent
storage_available(s::SerializationDiskStorage) = storage_available(s.parent)
storage_capacity(s::SerializationDiskStorage) = storage_capacity(s.parent)
