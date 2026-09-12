args <- commandArgs(trailingOnly = TRUE)
if (!length(args)) {
  stop("Usage: Rscript scripts/check.R models/<id> [--reference]", call. = FALSE)
}
model_arg <- args[!startsWith(args, "--")][1]
run_reference <- "--reference" %in% args
model_dir <- normalizePath(model_arg, mustWork = TRUE)
manifest <- file.path(model_dir, "kernel", "Cargo.toml")

cargo_lines <- readLines(manifest, warn = FALSE)
package_start <- match("[package]", trimws(cargo_lines), nomatch = 0L)
if (!package_start) stop("Cargo.toml has no [package] section", call. = FALSE)
after_package <- cargo_lines[seq.int(package_start + 1L, length(cargo_lines))]
next_section <- which(startsWith(trimws(after_package), "["))[1]
if (!is.na(next_section)) after_package <- head(after_package, next_section - 1L)
name_line <- grep("^[[:space:]]*name[[:space:]]*=", after_package, value = TRUE)[1]
crate <- sub(
  '^[[:space:]]*name[[:space:]]*=[[:space:]]*"([^"]+)".*$',
  "\\1",
  name_line
)

status <- system2(
  "cargo",
  c("build", "--release", "--locked", "--manifest-path", shQuote(manifest))
)
if (!identical(status, 0L)) stop("cargo build failed", call. = FALSE)

platform <- Sys.info()[["sysname"]]
if (.Platform$OS.type == "windows") {
  library_name <- paste0(gsub("-", "_", crate), ".dll")
} else if (identical(platform, "Darwin")) {
  library_name <- paste0("lib", gsub("-", "_", crate), ".dylib")
} else {
  library_name <- paste0("lib", gsub("-", "_", crate), ".so")
}
library_path <- normalizePath(
  file.path(model_dir, "kernel", "target", "release", library_name),
  mustWork = TRUE
)

data_path <- file.path(model_dir, "data.json")
if (!file.exists(data_path)) {
  stop("data.json is missing; run scripts/fetch-data.py first", call. = FALSE)
}
library(nutpieR)
reference <- nutpie_compile_model(file.path(model_dir, "model.stan"))
data <- jsonlite::fromJSON(data_path, simplifyVector = TRUE)
layout <- nutpie_kernel_layout(reference, data)
cat(sprintf("Layout: %d unconstrained coordinates\n", layout$ndim))
bound <- nutpie_attach_kernel(reference, library_path, data = data)
random <- nutpie_validate_kernel(bound)
print(random)
if (!identical(random$status, "pass")) stop("default kernel check failed", call. = FALSE)

if (run_reference) {
  reference_check <- nutpie_validate_kernel(bound, method = "reference")
  print(reference_check)
  if (!identical(reference_check$status, "pass")) {
    stop("reference kernel check failed", call. = FALSE)
  }
}
