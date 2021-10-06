@static if Sys.isunix()

struct StatFSStruct
    f_bsize::Culong
    f_frsize::Culong
    f_blocks::Culong
    f_bfree::Culong
    f_bavail::Culong
    f_files::Culong
    f_ffree::Culong
    f_favail::Culong
    f_fsid::Culong
    f_flag::Culong
    f_namemax::Culong
    reserved::NTuple{6,Cint}
end

function statvfs(path::String)
    buf = Ref{StatFSStruct}()
    GC.@preserve buf begin
        errno = ccall(:statvfs, Cint, (Cstring, Ptr{StatFSStruct}), path, buf)
        if errno != 0
            Base.systemerror(errno)
        end
        buf[]
    end
end

else

error("Not implemented yet")

end
