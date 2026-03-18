# Project Instructions

## Auto-update README

After implementing any new feature or significant change, automatically update README.md in the background to reflect the change. Use a background agent for this so it doesn't block the main workflow.

## Deployment

The phone is accessible via `ssh termux-phone`. After code changes:
1. `scp` changed files to `termux-phone:~/spare-paw/...`
2. `ssh termux-phone "pkill -f 'spare_paw'"` to stop
3. Restart: `ssh termux-phone "cd ~/spare-paw && nohup python -m spare_paw gateway > /data/data/com.termux/files/usr/tmp/spare-paw.log 2>&1 &"`
4. Check logs: `ssh termux-phone "tail -N /data/data/com.termux/files/usr/tmp/spare-paw.log | grep -v 'getUpdates'"`

## Prompt files

The bot loads personality/context from `~/.spare-paw/` on every turn:
- `IDENTITY.md` — bot personality
- `USER.md` — user preferences
- `SYSTEM.md` — device capabilities and behavior rules
