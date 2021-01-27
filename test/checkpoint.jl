@testset "Scheduler Checkpointing" begin
    ctx = Context([1,workers()...])
    d = Ref(false)
    a = delayed(()->begin
        d[] = true
        1
    end)()
    f = Ref(0)
    opts = Dagger.Sch.SchedulerOptions(;
    single=1,
    checkpoint=(result)->begin
        f[] = collect(result)
    end)
    @test collect(ctx, a; options=opts) == 1
    @test d[]
    @test f[] == 1

    opts = Dagger.Sch.SchedulerOptions(;
    single=1,
    restore=()->begin
        Dagger.tochunk(f[])
    end)
    f[] = 2
    d[] = false
    @test collect(ctx, a; options=opts) == 2
    @test !(d[]) # thunks aren't run again
    @test f[] == 2 # checkpoint isn't regenerated

    @testset "Checkpoint Failure" begin
        d = Ref(false)
        a = delayed(()->begin
            d[] = true
            1
        end)()
        e = Ref(false)
        opts = Dagger.Sch.SchedulerOptions(;
        single=1,
        checkpoint=(result)->begin
            e[] = true
            error("Test")
        end)
        @test collect(ctx, a; options=opts) == 1
        @test d[] # thunks executed
        @test e[] # checkpoint executed
    end
    @testset "Restore Failure" begin
        d = Ref(false)
        a = delayed(()->begin
            d[] = true
            1
        end)()
        e = Ref(false)
        opts = Dagger.Sch.SchedulerOptions(;
        single=1,
        restore=()->begin
            e[] = true
            error("Test")
        end)
        @test collect(ctx, a; options=opts) == 1
        @test d[] # thunks executed
        @test e[] # restore executed
    end
end

@testset "Thunk Checkpointing" begin
    ctx = Context([1,workers()...])
    d = Ref(false)
    opts = Dagger.Sch.ThunkOptions(;
    single=1,
    checkpoint=(thunk, result)->begin
        @assert thunk.f != Base.:*
        f[] = collect(result)
    end)
    a = delayed(()->begin
        d[] = true
        1
    end; options=opts)()
    b = delayed(*)(a,2)
    f = Ref(0)
    @test collect(ctx, b) == 2
    @test d[]
    @test f[] == 1

    opts = Dagger.Sch.ThunkOptions(;
    single=1,
    restore=(thunk)->begin
        @assert thunk.f != Base.:*
        Dagger.tochunk(f[])
    end)
    a = delayed(()->begin
        d[] = true
        1
    end; options=opts)()
    b = delayed(*)(a,2)
    f[] = 2
    d[] = false
    @test collect(ctx, b) == 4 # thunk b is run
    @test !(d[]) # thunk a isn't run again
    @test f[] == 2 # checkpoint isn't regenerated

    @testset "Checkpoint Failure" begin
        d = Ref(false)
        e = Ref(false)
        opts = Dagger.Sch.ThunkOptions(;
        single=1,
        checkpoint=(thunk, result)->begin
            e[] = true
            error("Test")
        end)
        a = delayed(()->begin
            d[] = true
            1
        end; options=opts)()
        @test collect(ctx, a) == 1
        @test d[] # thunk executed
        @test e[] # checkpoint executed
    end
    @testset "Restore Failure" begin
        d = Ref(false)
        e = Ref(false)
        opts = Dagger.Sch.ThunkOptions(;
        single=1,
        restore=(thunk)->begin
            e[] = true
            error("Test")
        end)
        a = delayed(()->begin
            d[] = true
            1
        end; options=opts)()
        @test collect(ctx, a) == 1
        @test d[] # thunk executed
        @test e[] # restore executed
    end
end
