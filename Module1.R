args <- commandArgs(trailingOnly = TRUE)
print(args)
# Argument one is going to be the directory.
# Argument two is going to be the file name.
# Argument three is going to be the output file name.
# Argument four is going to be the specific function to be called.

if (args[4]=="1"){

  result <-paste(".",args[1],paste(args[3],toString(".het"), sep = ""),sep="//")
  dat <- read.table(result, header=T) # Read in the EUR.het file, specify it has header
  m <- mean(dat$F) # Calculate the mean  
  s <- sd(dat$F) # Calculate the SD
  valid <- subset(dat, F <= m+3*s & F >= m-3*s) # Get any samples with F coefficient within 3 SD of the population mean
  result <-paste("./",args[1],paste(args[2],toString(".valid.sample"), sep = ""),sep="//")
  write.table(valid[,c(1,2)], result, quote=F, row.names=F) # print FID and IID for valid samples
  result <-paste("./",args[1],paste(args[2],toString(".bim"), sep = ""),sep="//")
  bim <- read.table(result)
  
  colnames(bim) <- c("CHR", "SNP", "CM", "BP", "B.A1", "B.A2")
  
  # Read in QCed SNPs
  result <-paste("./",args[1],paste(args[3],toString(".snplist"), sep = ""),sep="//")
  
  qc <- read.table(result, header = F, stringsAsFactors = F)
  # Read in the GWAS data
  path_parts <- strsplit(args[1], "/|\\\\")[[1]]
  print(path_parts)
  print(path_parts[1])
  result <-paste("./",path_parts[1],path_parts[2],paste(path_parts[1],toString(".gz"), sep = ""),sep="//")
  #result <-paste("./",args[1],"train",toString("train.assoc.fisher"),sep="//")
  
  height <-read.table(gzfile(result),
               header = T,
               stringsAsFactors = F, 
               sep="\t")
  # Change all alleles to upper case for easy comparison
  

  height$A1 <- toupper(height$A1)
  height$A2 <- toupper(height$A2)
  bim$B.A1 <- toupper(bim$B.A1)
  bim$B.A2 <- toupper(bim$B.A2)
  info <- merge(bim, height, by = c("SNP", "CHR", "BP"))
  # Filter QCed SNPs
  print(length(info))

  info <- info[info$SNP %in% qc$V1,]
  print(length(info))
  # Function for finding the complementary allele
  
  complement <- function(x) {
    switch (
      x,
      "A" = "T",
      "C" = "G",
      "T" = "A",
      "G" = "C",
      return(NA)
    )
  }
  
  # Get SNPs that have the same alleles across base and target
  info.match <- subset(info, A1 == B.A1 & A2 == B.A2)
  print(length(info.match))
   
  # Identify SNPs that are complementary between base and target
  info$C.A1 <- sapply(info$B.A1, complement)
  info$C.A2 <- sapply(info$B.A2, complement)
  info.complement <- subset(info, A1 == C.A1 & A2 == C.A2)
  # Update the complementary alleles in the bim file
  # This allow us to match the allele in subsequent analysis
  
  complement.snps <- bim$SNP %in% info.complement$SNP
  bim[complement.snps,]$B.A1 <-
    sapply(bim[complement.snps,]$B.A1, complement)
  bim[complement.snps,]$B.A2 <-
    sapply(bim[complement.snps,]$B.A2, complement)
  
  # identify SNPs that need recoding
  info.recode <- subset(info, A1 == B.A2 & A2 == B.A1)
  # Update the recode SNPs
  recode.snps <- bim$SNP %in% info.recode$SNP
  tmp <- bim[recode.snps,]$B.A1
  bim[recode.snps,]$B.A1 <- bim[recode.snps,]$B.A2
  bim[recode.snps,]$B.A2 <- tmp
  
  # identify SNPs that need recoding & complement
  info.crecode <- subset(info, A1 == C.A2 & A2 == C.A1)
  # Update the recode + strand flip SNPs
  com.snps <- bim$SNP %in% info.crecode$SNP
  tmp <- bim[com.snps,]$B.A1
  bim[com.snps,]$B.A1 <- as.character(sapply(bim[com.snps,]$B.A2, complement))
  bim[com.snps,]$B.A2 <- as.character(sapply(tmp, complement))
  result <-paste("./",args[1],paste(args[2],toString(".a1"), sep = ""),sep="//")
  
  # Output updated bim file
  write.table(
    bim[,c("SNP", "B.A1")],
    result,
    quote = F,
    row.names = F,
    col.names = F,
    sep="\t"
  )
  mismatch <-
    bim$SNP[!(bim$SNP %in% info.match$SNP |
                bim$SNP %in% info.complement$SNP | 
                bim$SNP %in% info.recode$SNP |
                bim$SNP %in% info.crecode$SNP)]
  result <-paste("./",args[1],paste(args[2],toString(".mismatch"), sep = ""),sep="//")
  
  write.table(
    mismatch,
    result,
    quote = F,
    row.names = F,
    col.names = F
  )
  
  
}
if (args[4]=="2"){
  result <-paste("./",args[1],paste(args[2],toString(".valid.sample"), sep = ""),sep="//")
  valid <- read.table(result, header=T)
  result <-paste("./",args[1],paste(args[3],toString(".sexcheck"), sep = ""),sep="//")
  dat <- read.table(result, header=T)
  valid <- subset(dat, STATUS=="OK" & FID %in% valid$FID)
  result <-paste("./",args[1],paste(args[3],toString(".valid"), sep = ""),sep="//")
  print(result)
  write.table(valid[,c("FID", "IID")], result, row.names=F, col.names=F, sep="\t", quote=F) 
}
