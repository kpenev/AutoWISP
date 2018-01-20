# Crash  Recovery {#CrashRecovery_page}

It is imperative that the pipeline is robust against potential crashes at any
point during processing. Here we outline the mechanism (to be) used to
accomplish this.

Dangerous situations arising due to a crash in the middle of an operation that:
  1. creates or updates a file
  2. must create/update multiple files in a consistent way
  3. creates/updates file(s) and marks that in the database
  4. must perform several database modifications in a consistent way 
