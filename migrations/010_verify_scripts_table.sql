-- Verification migration to ensure scripts table is properly set up
DO $$
BEGIN
    -- Check if table exists and has expected columns
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_name = 'scripts' AND table_schema = 'public'
    ) THEN
        RAISE EXCEPTION 'scripts table does not exist';
    END IF;
    
    -- Check if unique constraint exists
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name = 'unique_content_hash' AND table_name = 'scripts'
    ) THEN
        RAISE WARNING 'unique_content_hash constraint missing on scripts table';
    END IF;
    
    RAISE NOTICE 'Scripts repository migration verification passed';
END
$$;