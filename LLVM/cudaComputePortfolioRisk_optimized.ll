; ModuleID = 'cudaComputePortfolioRisk.ll'
source_filename = "cudaComputePortfolioRisk.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite)
define dso_local void @_Z14matrixMultiplyPdS_S_iii(ptr nocapture noundef readonly %A, ptr nocapture noundef readonly %B, ptr nocapture noundef writeonly %C, i32 noundef %ARows, i32 noundef %ACols, i32 noundef %BCols) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %mul = mul i32 %0, %1
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %add = add i32 %mul, %2
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %mul5 = mul i32 %3, %4
  %5 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %add7 = add i32 %mul5, %5
  %cmp = icmp slt i32 %add, %ARows
  %cmp8 = icmp slt i32 %add7, %BCols
  %or.cond = and i1 %cmp, %cmp8
  br i1 %or.cond, label %for.cond.preheader, label %if.end

for.cond.preheader:                               ; preds = %entry
  %cmp933 = icmp sgt i32 %ACols, 0
  br i1 %cmp933, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %for.cond.preheader
  %mul10 = mul nsw i32 %add, %ACols
  %xtraiter = and i32 %ACols, 3
  %6 = icmp ult i32 %ACols, 4
  br i1 %6, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body.lr.ph.new

for.body.lr.ph.new:                               ; preds = %for.body.lr.ph
  %unroll_iter = and i32 %ACols, -4
  %7 = and i32 %ACols, -4
  br label %for.body

for.cond.cleanup.loopexit.unr-lcssa:              ; preds = %for.body, %for.body.lr.ph
  %add17.lcssa.ph = phi double [ undef, %for.body.lr.ph ], [ %add17.3, %for.body ]
  %k.035.unr = phi i32 [ 0, %for.body.lr.ph ], [ %7, %for.body ]
  %sum.034.unr = phi double [ 0.000000e+00, %for.body.lr.ph ], [ %add17.3, %for.body ]
  %lcmp.mod.not = icmp eq i32 %xtraiter, 0
  br i1 %lcmp.mod.not, label %for.cond.cleanup, label %for.body.epil

for.body.epil:                                    ; preds = %for.cond.cleanup.loopexit.unr-lcssa, %for.body.epil
  %k.035.epil = phi i32 [ %inc.epil, %for.body.epil ], [ %k.035.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %sum.034.epil = phi double [ %add17.epil, %for.body.epil ], [ %sum.034.unr, %for.cond.cleanup.loopexit.unr-lcssa ]
  %epil.iter = phi i32 [ %epil.iter.next, %for.body.epil ], [ 0, %for.cond.cleanup.loopexit.unr-lcssa ]
  %add11.epil = add nsw i32 %k.035.epil, %mul10
  %idxprom.epil = sext i32 %add11.epil to i64
  %arrayidx.epil = getelementptr inbounds double, ptr %A, i64 %idxprom.epil
  %8 = load double, ptr %arrayidx.epil, align 8, !tbaa !8
  %mul12.epil = mul nsw i32 %k.035.epil, %BCols
  %add13.epil = add nsw i32 %mul12.epil, %add7
  %idxprom14.epil = sext i32 %add13.epil to i64
  %arrayidx15.epil = getelementptr inbounds double, ptr %B, i64 %idxprom14.epil
  %9 = load double, ptr %arrayidx15.epil, align 8, !tbaa !8
  %mul16.epil = fmul contract double %8, %9
  %add17.epil = fadd contract double %sum.034.epil, %mul16.epil
  %inc.epil = add nuw nsw i32 %k.035.epil, 1
  %epil.iter.next = add nuw nsw i32 %epil.iter, 1
  %epil.iter.cmp.not = icmp eq i32 %epil.iter.next, %xtraiter
  br i1 %epil.iter.cmp.not, label %for.cond.cleanup, label %for.body.epil, !llvm.loop !12

for.cond.cleanup:                                 ; preds = %for.body.epil, %for.cond.cleanup.loopexit.unr-lcssa, %for.cond.preheader
  %sum.0.lcssa = phi double [ 0.000000e+00, %for.cond.preheader ], [ %add17.lcssa.ph, %for.cond.cleanup.loopexit.unr-lcssa ], [ %add17.epil, %for.body.epil ]
  %mul18 = mul nsw i32 %add, %BCols
  %add19 = add nsw i32 %mul18, %add7
  %idxprom20 = sext i32 %add19 to i64
  %arrayidx21 = getelementptr inbounds double, ptr %C, i64 %idxprom20
  store double %sum.0.lcssa, ptr %arrayidx21, align 8, !tbaa !8
  br label %if.end

for.body:                                         ; preds = %for.body, %for.body.lr.ph.new
  %k.035 = phi i32 [ 0, %for.body.lr.ph.new ], [ %inc.3, %for.body ]
  %sum.034 = phi double [ 0.000000e+00, %for.body.lr.ph.new ], [ %add17.3, %for.body ]
  %add11 = add nsw i32 %k.035, %mul10
  %idxprom = sext i32 %add11 to i64
  %arrayidx = getelementptr inbounds double, ptr %A, i64 %idxprom
  %10 = load double, ptr %arrayidx, align 8, !tbaa !8
  %mul12 = mul nsw i32 %k.035, %BCols
  %add13 = add nsw i32 %mul12, %add7
  %idxprom14 = sext i32 %add13 to i64
  %arrayidx15 = getelementptr inbounds double, ptr %B, i64 %idxprom14
  %11 = load double, ptr %arrayidx15, align 8, !tbaa !8
  %mul16 = fmul contract double %10, %11
  %add17 = fadd contract double %sum.034, %mul16
  %inc = or i32 %k.035, 1
  %add11.1 = add nsw i32 %inc, %mul10
  %idxprom.1 = sext i32 %add11.1 to i64
  %arrayidx.1 = getelementptr inbounds double, ptr %A, i64 %idxprom.1
  %12 = load double, ptr %arrayidx.1, align 8, !tbaa !8
  %mul12.1 = mul nsw i32 %inc, %BCols
  %add13.1 = add nsw i32 %mul12.1, %add7
  %idxprom14.1 = sext i32 %add13.1 to i64
  %arrayidx15.1 = getelementptr inbounds double, ptr %B, i64 %idxprom14.1
  %13 = load double, ptr %arrayidx15.1, align 8, !tbaa !8
  %mul16.1 = fmul contract double %12, %13
  %add17.1 = fadd contract double %add17, %mul16.1
  %inc.1 = or i32 %k.035, 2
  %add11.2 = add nsw i32 %inc.1, %mul10
  %idxprom.2 = sext i32 %add11.2 to i64
  %arrayidx.2 = getelementptr inbounds double, ptr %A, i64 %idxprom.2
  %14 = load double, ptr %arrayidx.2, align 8, !tbaa !8
  %mul12.2 = mul nsw i32 %inc.1, %BCols
  %add13.2 = add nsw i32 %mul12.2, %add7
  %idxprom14.2 = sext i32 %add13.2 to i64
  %arrayidx15.2 = getelementptr inbounds double, ptr %B, i64 %idxprom14.2
  %15 = load double, ptr %arrayidx15.2, align 8, !tbaa !8
  %mul16.2 = fmul contract double %14, %15
  %add17.2 = fadd contract double %add17.1, %mul16.2
  %inc.2 = or i32 %k.035, 3
  %add11.3 = add nsw i32 %inc.2, %mul10
  %idxprom.3 = sext i32 %add11.3 to i64
  %arrayidx.3 = getelementptr inbounds double, ptr %A, i64 %idxprom.3
  %16 = load double, ptr %arrayidx.3, align 8, !tbaa !8
  %mul12.3 = mul nsw i32 %inc.2, %BCols
  %add13.3 = add nsw i32 %mul12.3, %add7
  %idxprom14.3 = sext i32 %add13.3 to i64
  %arrayidx15.3 = getelementptr inbounds double, ptr %B, i64 %idxprom14.3
  %17 = load double, ptr %arrayidx15.3, align 8, !tbaa !8
  %mul16.3 = fmul contract double %16, %17
  %add17.3 = fadd contract double %add17.2, %mul16.3
  %inc.3 = add nuw nsw i32 %k.035, 4
  %niter.ncmp.3 = icmp eq i32 %inc.3, %unroll_iter
  br i1 %niter.ncmp.3, label %for.cond.cleanup.loopexit.unr-lcssa, label %for.body, !llvm.loop !14

if.end:                                           ; preds = %for.cond.cleanup, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.y() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #1

attributes #0 = { mustprogress nofree norecurse nosync nounwind memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_75" "target-features"="+ptx81,+sm_75" "uniform-work-group-size"="true" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2, !3}
!nvvm.annotations = !{!4}
!llvm.ident = !{!5, !6}
!nvvmir.version = !{!7}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 1]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{ptr @_Z14matrixMultiplyPdS_S_iii, !"kernel", i32 1}
!5 = !{!"clang version 18.0.0 (https://github.com/llvm/llvm-project.git 26eb4285b56edd8c897642078d91f16ff0fd3472)"}
!6 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!7 = !{i32 2, i32 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"double", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.unroll.disable"}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
