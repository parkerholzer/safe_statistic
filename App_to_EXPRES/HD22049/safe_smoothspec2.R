library(parallel)
filenames = list.files(path = "22049spec", pattern = "*_2.csv")
options(digits=10)
srtid = as.numeric(gsub("ctd_2.csv", "", gsub("22049_", "", filenames)))
filenames = filenames[order(srtid)]

SPECTRA = mclapply(filenames, function(f) read.csv(paste0("22049spec/",f)), mc.cores=19)

print("Read in all spectra")

for(i in 1:length(SPECTRA)){
  SPECTRA[[i]]$Flux = SPECTRA[[i]]$Flux/SPECTRA[[i]]$Continuum
}

print("Made it here")

wvl = c()
flx = c()
for(spec in SPECTRA){
  keep = which((spec$Wavelength >= 4400) & 
                 (spec$Wavelength <= 7000) &
                 (spec$Flux > 0) & 
                 (!is.na(spec$Flux)))
  wvl = c(wvl, spec$Wavelength[keep])
  flx = c(flx, spec$Flux[keep])
}

flx = flx[order(wvl)]
wvl = wvl[order(wvl)]

jumps = which(wvl[2:length(wvl)] - 
                wvl[1:(length(wvl)-1)] > 3*0.017)

library(locfit)


lbs = c(wvl[1], wvl[jumps+1])
ubs = c(wvl[jumps], wvl[length(wvl)]) + rep_len(1e-8,length(jumps)+1)
lbs2 = c()
for(i in 1:length(lbs)){
  if(ubs[i] - lbs[i] > 8){
    brks = seq(lbs[i], ubs[i], 4)
    if(ubs[i] - brks[length(brks)] < 3){
      brks = brks[1:(length(brks)-1)]
    }
    lbs2 = c(lbs2, brks[2:length(brks)])
  }else if(ubs[i] - lbs[i] >= 6){
    lbs2 = c(lbs2, lbs[i] + (ubs[i] - lbs[i])/2)
  }
}
lbs = sort(c(lbs, lbs2))
ubs = sort(c(ubs, lbs2))
bnds = mclapply(1:length(lbs), function(i) c(lbs[i], ubs[i]), mc.cores = 19)

print("Made it here 2")

smoothspec = function(bd){
  keep = which((wvl >= bd[1]) & (wvl < bd[2]))
  predwvl = seq(wvl[keep][1], wvl[keep][length(keep)],
                length.out = as.integer(3*length(keep)/length(SPECTRA)))
  if(length(keep) < 100){
     return(list(predwvl, rep_len(NA, length(predwvl))))
  }
  amin = length(which(wvl[keep] <= wvl[keep][1] + 0.017))/length(keep)
  amax = length(which(wvl[keep] <= wvl[keep][1] + 0.05))/length(keep)
  alphas = seq(amin, amax, length.out = 20)
  gcvs = gcvplot(flx[keep] ~ wvl[keep],
                 deg=2, alpha=alphas, 
                 kern='gauss')
  bestalpha = gcvs$alpha[which.min(gcvs$values)]
  mdl = locfit(flx[keep] ~ wvl[keep], deg=2, 
                      alpha=bestalpha, kern='gauss')
  
  return(list(predwvl, predict(mdl, predwvl)))
}

fittedflx = mclapply(bnds, smoothspec, mc.cores=19)

wvl = unlist(mclapply(fittedflx, function(lst) lst[[1]], mc.cores = 19))
flx = unlist(mclapply(fittedflx, function(lst) lst[[2]], mc.cores = 19))

print("Made it here 3")

keep = which(!is.na(as.numeric(wvl)))

smoothtemp = data.frame(Wavelength = wvl[keep], Flux = flx[keep])
write.csv(smoothtemp, file = "22049smoothtemp2.csv", row.names=FALSE)
