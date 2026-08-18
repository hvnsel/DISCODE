% data_combine.m
S_LO = load("AllData_ProcessedLOhit.mat");
S_NO = load("AllData_ProcessedNOhit.mat");

S_combined = struct();

S_combined.Acc = cat(3, S_LO.Data.Acc, S_NO.Data.Acc);
S_combined.Acc2 = cat(3, S_LO.Data.Acc2, S_NO.Data.Acc2);

S_combined.Vel = cat(3, S_LO.Data.Vel, S_NO.Data.Vel);
S_combined.Vel2 = cat(3, S_LO.Data.Vel2, S_NO.Data.Vel2);

S_combined.Disp = cat(3, S_LO.Data.Disp, S_NO.Data.Disp);
S_combined.Disp2 = cat(3, S_LO.Data.Disp2, S_NO.Data.Disp2);

S_combined.Time = S_LO.Data.Time
Data = S_combined
save("LONO_combined.mat", "Data");

