"""Unknown slash commands suggest close skill commands (Task 7, M1).

Brief's test, adapted to the real cli.py surface:
  * the class is ``HermesCLI`` (no ``HermesConsole`` exists in cli.py);
  * ``_expand_slash_prefix`` takes ``(cmd_original, cmd_lower, skill_commands,
    skill_bundles)`` — pass empty tables so the typo lands on the unknown branch;
  * ``HERMES_HOME`` is set BEFORE ``clear_skills_system_prompt_cache(clear_snapshot=True)``
    (clear_snapshot unlinks ``<home>/.skills_prompt_snapshot.json`` — clearing first
    would delete the developer's real snapshot; same ordering as the Task 6 harness).
"""
from unittest.mock import patch


class TestSlashSuggest:
    def test_unknown_slash_suggests_skill_command(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from agent.prompt_builder import clear_skills_system_prompt_cache
        clear_skills_system_prompt_cache(clear_snapshot=True)
        d = tmp_path / "skills"; d.mkdir()
        (d / "web-search").mkdir()
        (d / "web-search" / "SKILL.md").write_text("---\nname: web-search\ndescription: Search the web.\n---\nbody\n")
        import tools.skills_tool as st
        from tools.skills_tool import _reset_skill_search_cache
        _reset_skill_search_cache()
        with patch.object(st, "SKILLS_DIR", tmp_path / "skills"):
            from cli import HermesCLI
            console = HermesCLI.__new__(HermesCLI)   # no full init
            console._expand_slash_prefix("web-serach", "web-serach", {}, {})
        out = capsys.readouterr().out
        assert "Unknown command" in out
        assert "/web-search" in out
