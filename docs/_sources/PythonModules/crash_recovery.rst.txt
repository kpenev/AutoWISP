***************
Crash  Recovery
***************

It is imperative that the pipeline is robust against potential crashes at any
point during processing. Here we outline the mechanism (to be) used to
accomplish this.

Dangerous situations arising due to a crash in the middle of an operation that:

    1. creates or updates a file

    2. must create/update multiple files in a consistent way

    3. creates/updates file(s) and marks that in the database

    4. must perform several database modifications in a consistent way 

The recovery system maintains recovery information in a database table, named 
``RecoveryInformation`` as well as original copies of files being updated in a 
temporary directory. 

The table gets updated before any risky operation starts with information on how
to restore the filesystem to the state before the operation begins. Database 
updates are performed only at the very end of the operation. The update is 
performed through a database session which combines  that with the removing of 
the relevant recovery information from ``RecoveryInformation``.

In order to allow recovery from both new files and file updates, the
``RecoveryInformation`` maintains a list of updated files as well a copy of the
original file in a temporary location. For new files the original file entry is
``NULL``. 

The order of operations is as follows:

    1. The original files are copied to the temporary location

    2. The list of final and original copies is added to the database in a single
       session.
    3. The operation is performed

    4. The databese is updated as required by the operation and to delete the
       entries in ``RecoveryInformation`` created during step 1.

    5. The temporary copies of the original files are deleted.
  
Upon starting of the pipeline recovery proceeds as follows:

    1. Any file in the temporary directory not listed in the
       ``RecoveryInformation`` table are deleted.

    2. Any updated files listed in ``RecoveryInformation`` are overwritten by
       their originals.

    3. The ``RecoveryInformation`` table is emptied.

    4. The temporary directory is emptied.
  
This way if interruption occurs during:

    * step 1: any copies of original files made will be deleted during recovery
      step 1 and the state of the database and file system will be exactly as it
      was before the operation started.

    * step 2: the database will be automatically rolled back by the database and
      again file system and database will be in their original states.

    * step 3: any updates to the file system will be undone by recovery step 2
      and step 3 will restore the database.

    * step 4: The database update will be rolled back by the database and we
      will be back to the case above.

    * step 5: the database will already have the final state for the operation
      and recovery step 1 will take the file system to the desired final state.
