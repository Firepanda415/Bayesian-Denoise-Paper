################################################################################
# File: MA223Styles
# Course: MA223 Engineering Statistics I
# Description: Additional R functions for creating MA223 styles.
#
# Author: Eric Reyes
# Date: Fall 2017

## ---- Load Packages ----
pkgs <- c("tidyverse",
          "HDInterval",
          "rstan",
          "bridgesampling",
          "knitr")

for(pkg in pkgs) library(pkg, character.only = TRUE)


## ---- Change Options ----
# Suppress status bar in dplyr.
options(dplyr.show_progress = FALSE)

# Change theme for plots
theme_set(theme_bw(12))
theme_update(legend.position = "bottom")

# Specify chunck options
knitr::opts_chunk$set(
  prompt = FALSE,
  comment = "",
  message = FALSE,
  warning = FALSE)


## ---- Create Special Blocks ----
eng_instructor <- function(options) {
  if (identical(options$echo, FALSE)) return()
  
  # Steal some ideas from block definition
  to = opts_knit$get("rmarkdown.pandoc.to")
  is_pandoc = !is.null(to)
  if(!is_pandoc){
    to = opts_knit$get("out.format")
    if(!(to %in% c("latex", "html", "markdown"))) to = NULL
  }
  
  if(is.null(to)) return(code)
  if(to=="beamer") to = "latex"
  if(grepl("(markdown)|(epub)|(html)|(revealjs)|(s5)|(slideous)|(slidy)",
           to)) to = "html"
  
  
  code = paste(options$code, collapse = '\n'); type = options$type
  if (is.null(type)) return(code)
  
  if(!is.null(type) && type=="solution"){
    code = paste("__SOLUTION__:", code)
  }
  
  if (is.null(opts_knit$get("rmarkdown.pandoc.to"))) stop('The engine "block2" is for R Markdown only')
  
  l1 = options$latex.options
  if (is.null(l1)) l1 = ''
  # protect environment options because Pandoc may escape the characters like
  # {}; when encoded in integers, they won't be escaped, but will need to
  # restore them later; see bookdown:::restore_block2
  if (l1 != '') l1 = paste(
    c('\\iffalse{', utf8ToInt(enc2utf8(l1)), '}\\fi{}'), collapse = '-'
  )
  h2 = ifelse(is.null(options$html.tag), 'div', options$html.tag)
  h3 = ifelse(is.null(options$html.before), '', options$html.before)
  h4 = ifelse(is.null(options$html.after), '', options$html.after)
  h5 = ifelse(is.null(options$html.before2), '', options$html.before2)
  h6 = ifelse(is.null(options$html.after2), '', options$html.after2)
  
  if(to=="latex"){
    sprintf('\\BeginKnitrBlock{%s}%s\n%s%s%s\n\\EndKnitrBlock{%s}',
            type, l1, h5, code, h6, type)
  } else {
    sprintf(
      '\\BeginKnitrBlock{%s}%s%s<%s class="%s" custom-style="%s">%s%s%s</%s>%s\\EndKnitrBlock{%s}',
      type, l1, h3, h2, type, type, h5, code, h6, h2, h4, type
    )
  }
}


knit_engines$set(c(knit_engines$get(), 
                   "instructor" = eng_instructor))


# Special Functions

# function: stan_to_df
# description: Convert the parameter arguments from stan to a data frame.
#
# Parameters:
#   object     stanfit object (see extract()).
#   ...        additional parameters to pass to extract().
stan_to_df <- function(object, ...){
  params <- rstan::extract(object, ...)
  
  if(class(params)=="list"){
    if(any(sapply(lapply(params, dim), length) > 1)){
     dims <- params %>%
       sapply(function(u) ifelse(length(dim(u))>1, dim(u)[2], 1))
     
     col.names <- names(dims) %>%
       sapply(function(u){
         if(dims[u]>1){
           paste(u, seq(dims[u]), sep = "")
         } else {
           u
         }
       }) %>%
       unlist()
     
     params <- params %>%
       lapply(as_data_frame) %>%
       do.call(what = "cbind")
     
     colnames(params) <- col.names
     
     params %>% as_data_frame()
    } else {
      params %>%
        as_data_frame() %>%
        mutate(`.iteration` = seq(nrow(.)))
    }
  } else if(class(params)=="array"){
    col.names <- dimnames(params)$parameters
    
    params <- params %>%
      apply(3, function(u){
        data_frame(`_Value` = c(u),
                   `_Chain` = rep(seq_along(dimnames(u)[[2]]),
                               each = nrow(u)),
                   `_Iteration` = rep(seq(nrow(u)),
                                      times = length(dimnames(u)[[2]])))
      }) %>%
      do.call(what = "cbind")
    
    params <- params %>%
      select(contains("_Value")) %>%
      cbind(factor(params[,ncol(params)-1]),
            params[,ncol(params)])
    
    colnames(params) <- gsub(pattern = "\\[|\\]", replacement = "", 
                             x = c(col.names, ".chain", ".iteration"))
    
    params %>% as_data_frame()
  }
}
