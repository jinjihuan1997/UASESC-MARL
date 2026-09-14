from support import *

status=read(HERE/'status.json')
status.update(state='INTERRUPTED_OR_FAILED',exit_code=int(sys.argv[1]),updated_utc=stamp(),
              note='Only hash-verified complete batches/epochs can be resumed; no automatic changes to settings.')
write(HERE/'status.json',status)
with (HERE/'logs/failures.jsonl').open('a') as f:
    f.write(json.dumps(status,ensure_ascii=False)+'\n')
