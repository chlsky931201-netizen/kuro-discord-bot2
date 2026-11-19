import base64
import cv2
import json
import numpy as np
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import zipfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi import HTTPException
import discord
from core.auth_client import (
    AuthenticationError,
    LicenseMonitor,
    license_tier_allows,
    protect_local_secret,
    unprotect_local_secret,
)
from core.app_updater import UpdateError, _extract_package, update_state
from core.app_version import APP_VERSION
from core import equipment_apply_worker, equipment_window_controls, win_control
from core.equipment_window_controls import (
    close_equipment_window_if_open,
    ensure_equipment_window_open,
)
from core.equipment_removal_worker import EquipmentRemovalWorker
from core.multi_character_tester import MultiCharacterTester
from core.storage_worker import (
    STORAGE_MENU_DELAY_SECONDS,
    StorageWorker,
    build_storage_batches,
)
from pynput.keyboard import Key, KeyCode
from auth_server.bot import _is_admin, create_bot
from ui.main_window import KuroHelper

from auth_server.app import RedeemRequest, VerifyRequest, health_check, redeem, verify
from auth_server.database import (
    clear_app_update,
    get_app_update,
    get_pending_code,
    initialize_database,
    get_license_status,
    issue_code,
    reset_device,
    set_license_session_expiry,
    revoke_user,
    resolve_license_id,
    set_license_nickname,
    set_license_tier,
    set_app_update,
)
import ui.main_window as main_window


class AuthFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        environment = dict(os.environ)
        for name in ("SUPABASE_URL", "SUPABASE_KEY", "RENDER"):
            environment.pop(name, None)
        environment["AUTH_DB_PATH"] = str(Path(self.temp_dir.name) / "auth.sqlite3")
        environment["CODE_ENCRYPTION_KEY"] = base64.urlsafe_b64encode(
            b"0" * 32
        ).decode("ascii")
        environment_patch = patch.dict(os.environ, environment, clear=True)
        environment_patch.start()
        self.addCleanup(environment_patch.stop)
        for target in (
            "auth_server.database._get_supabase_client",
            "socket.socket.connect",
            "socket.create_connection",
        ):
            network_patch = patch(target, side_effect=AssertionError("External access during test"))
            network_patch.start()
            self.addCleanup(network_patch.stop)
        initialize_database()
        self.first_device = "device-test-0000000000001"
        self.second_device = "device-test-0000000000002"

    def test_code_is_one_time_and_session_is_device_bound(self):
        first_code, license_id = issue_code()
        token = redeem(
            RedeemRequest(
                code=first_code,
                device_id=self.first_device,
            )
        )["access_token"]
        result = verify(
            VerifyRequest(device_id=self.first_device),
            f"Bearer {token}",
        )
        self.assertTrue(result["authorized"])
        self.assertEqual(result["license_id"], license_id)

        with self.assertRaises(HTTPException) as other_device_error:
            verify(
                VerifyRequest(device_id=self.second_device),
                f"Bearer {token}",
            )
        self.assertEqual(other_device_error.exception.status_code, 401)

        with self.assertRaises(HTTPException) as reuse_error:
            redeem(
                RedeemRequest(
                    code=first_code,
                    device_id=self.first_device,
                )
            )
        self.assertEqual(reuse_error.exception.status_code, 401)

    def test_local_secret_is_encrypted_and_restored(self):
        secret = "multi-character-password"

        encrypted = protect_local_secret(secret)

        self.assertNotIn(secret, encrypted)
        self.assertEqual(unprotect_local_secret(encrypted), secret)

    def test_load_settings_keeps_encrypted_multi_character_password(self):
        helper = KuroHelper.__new__(KuroHelper)
        helper.settings_path = Path(self.temp_dir.name) / "settings.json"
        encrypted_password = protect_local_secret("multi-character-password")
        helper.settings_path.write_text(
            json.dumps({"multi_character_password_encrypted": encrypted_password}),
            encoding="utf-8",
        )

        settings = helper.load_settings()

        self.assertEqual(
            settings["multi_character_password_encrypted"],
            encrypted_password,
        )

    def test_load_settings_restores_equipment_slot_states(self):
        helper = KuroHelper.__new__(KuroHelper)
        helper.settings_path = Path(self.temp_dir.name) / "settings.json"
        helper.settings_path.write_text(
            json.dumps({"equipment_slot_states": {"무기": 1, "반지1": 2}}),
            encoding="utf-8",
        )

        settings = helper.load_settings()

        self.assertEqual(
            settings["equipment_slot_states"],
            {"무기": 1, "반지1": 2},
        )

    def test_equipment_slot_states_use_distinct_purple_and_gold_colors(self):
        button = unittest.mock.Mock()
        app = SimpleNamespace(
            equipment_slot_states={},
            equipment_slot_buttons={"무기": button},
        )

        KuroHelper._set_equipment_slot_state(app, "무기", 1)
        purple = button.configure.call_args.kwargs["fg_color"]
        KuroHelper._set_equipment_slot_state(app, "무기", 2)
        gold = button.configure.call_args.kwargs["fg_color"]

        self.assertEqual(purple, "#7044B8")
        self.assertEqual(gold, "#C99524")
        self.assertNotEqual(purple, gold)

    def test_visual_theme_does_not_reset_equipment_slot_button(self):
        button = SimpleNamespace(
            winfo_children=lambda: [],
            configure=unittest.mock.Mock(),
        )
        app = SimpleNamespace(equipment_slot_buttons={"무기": button})

        KuroHelper._apply_visual_theme(app, button)

        button.configure.assert_not_called()

    def test_reset_allows_rebinding_and_revoke_invalidates_session(self):
        first_code, license_id = issue_code()
        first_token = redeem(
            RedeemRequest(
                code=first_code,
                device_id=self.first_device,
            )
        )["access_token"]
        second_code = reset_device(license_id)
        second_token = redeem(
            RedeemRequest(
                code=second_code,
                device_id=self.second_device,
            )
        )["access_token"]
        self.assertTrue(
            verify(
                VerifyRequest(device_id=self.second_device),
                f"Bearer {second_token}",
            )["authorized"]
        )

        revoke_user(license_id)
        with self.assertRaises(HTTPException) as revoke_error:
            verify(
                VerifyRequest(device_id=self.second_device),
                f"Bearer {second_token}",
            )
        self.assertEqual(revoke_error.exception.status_code, 401)
        with self.assertRaises(HTTPException):
            verify(
                VerifyRequest(device_id=self.first_device),
                f"Bearer {first_token}",
            )

    def test_unregistered_code_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            redeem(
                RedeemRequest(
                    code="KURO-NOT-REGISTERED",
                    device_id=self.first_device,
                )
            )
        self.assertEqual(error.exception.status_code, 401)

    def test_codes_are_hashed_and_duplicates_are_rejected(self):
        code, _license_id = issue_code()
        with sqlite3.connect(os.environ["AUTH_DB_PATH"]) as connection:
            stored_hash = connection.execute(
                "SELECT code_hash FROM auth_codes"
            ).fetchone()[0]
        self.assertNotEqual(code, stored_hash)

    def test_verify_updates_online_presence_and_stales_after_90_seconds(self):
        code, license_id = issue_code()
        token = redeem(
            RedeemRequest(code=code, device_id=self.first_device)
        )["access_token"]

        self.assertTrue(get_license_status(license_id)["online"])
        verify(
            VerifyRequest(device_id=self.first_device),
            f"Bearer {token}",
        )
        status = get_license_status(license_id)
        self.assertTrue(status["online"])
        self.assertIsNotNone(status["last_seen_at"])

        with sqlite3.connect(os.environ["AUTH_DB_PATH"]) as connection:
            connection.execute(
                "UPDATE licenses SET last_seen_at = ? WHERE license_id = ?",
                (time.time() - 91, license_id),
            )
        self.assertFalse(get_license_status(license_id)["online"])

    def test_code_format_and_expiry_are_validated(self):
        code, license_id = issue_code()
        self.assertRegex(code, r"^(?:[A-HJ-NP-Z2-9]{4}-){5}[A-HJ-NP-Z2-9]{4}$")
        self.assertRegex(license_id, r"^[A-F0-9]{16}$")
        with self.assertRaises(ValueError):
            issue_code(expires_days=366)

    def test_pending_code_is_retrievable_by_id_or_nickname_only_before_use(self):
        code, license_id = issue_code(nickname="테스터 A")
        self.assertEqual(get_pending_code(license_id)["code"], code)
        self.assertEqual(get_pending_code("테스터 A")["license_id"], license_id)

        token = redeem(
            RedeemRequest(code=code, device_id=self.first_device)
        )["access_token"]
        with self.assertRaisesRegex(ValueError, "미사용 코드"):
            get_pending_code(license_id)
        with sqlite3.connect(os.environ["AUTH_DB_PATH"]) as connection:
            encrypted_code = connection.execute(
                "SELECT code_encrypted FROM auth_codes"
            ).fetchone()[0]
        self.assertIsNone(encrypted_code)
        self.assertTrue(token)

    def test_reset_device_reissues_retrievable_code(self):
        _old_code, license_id = issue_code(nickname="재발급 대상")
        new_code = reset_device(license_id)
        self.assertEqual(get_pending_code("재발급 대상")["code"], new_code)

    def test_set_license_session_expiry_updates_current_sessions(self):
        code, license_id = issue_code(expires_days=5)
        token = redeem(
            RedeemRequest(code=code, device_id=self.first_device)
        )["access_token"]

        self.assertEqual(set_license_session_expiry(license_id, 90), license_id)
        connection = sqlite3.connect(os.environ["AUTH_DB_PATH"])
        try:
            expires_at = connection.execute(
                "SELECT expires_at FROM sessions WHERE license_id = ?",
                (license_id,),
            ).fetchone()
        finally:
            connection.close()
        self.assertAlmostEqual(expires_at[0], time.time() + 90 * 24 * 60 * 60, delta=2)

        set_license_session_expiry(license_id, 0)
        connection = sqlite3.connect(os.environ["AUTH_DB_PATH"])
        try:
            expires_at = connection.execute(
                "SELECT expires_at FROM sessions WHERE license_id = ?",
                (license_id,),
            ).fetchone()[0]
        finally:
            connection.close()
        self.assertIsNone(expires_at)

        _unused_code, unused_license_id = issue_code()
        with self.assertRaisesRegex(ValueError, "인증 세션이 없습니다"):
            set_license_session_expiry(unused_license_id, 30)
        with self.assertRaises(ValueError):
            set_license_session_expiry(license_id, 366)
        self.assertTrue(
            verify(
                VerifyRequest(device_id=self.first_device),
                f"Bearer {token}",
            )["authorized"]
        )

    def test_license_management_accepts_unique_nickname(self):
        _code, license_id = issue_code(nickname="별명 사용자")
        self.assertEqual(resolve_license_id("별명 사용자"), license_id)
        self.assertEqual(
            set_license_tier("별명 사용자", "premium")["tier"],
            "premium",
        )
        set_license_nickname("별명 사용자", "새 별명")
        new_code = reset_device("새 별명")
        self.assertEqual(get_pending_code("새 별명")["code"], new_code)
        revoke_user("새 별명")
        self.assertFalse(get_license_status("새 별명")["active"])

        issue_code(nickname="중복 별명")
        issue_code(nickname="중복 별명")
        with self.assertRaisesRegex(ValueError, "같은 별명"):
            resolve_license_id("중복 별명")

    def test_developer_tier_issue_redeem_and_live_tier_changes(self):
        code, license_id = issue_code(tier="developer", nickname="개발용 테스터")
        result = redeem(RedeemRequest(code=code, device_id=self.first_device))
        self.assertEqual(result["tier"], "developer")
        self.assertTrue(license_tier_allows(result["tier"], "developer_utilities"))
        for tier in ("developer", "premium", "basic", "developer"):
            with self.subTest(tier=tier):
                self.assertEqual(set_license_tier(license_id, tier)["tier"], tier)
                verified = verify(VerifyRequest(device_id=self.first_device), f"Bearer {result['access_token']}")
                self.assertEqual(verified["tier"], tier)
                self.assertEqual(license_tier_allows(verified["tier"], "developer_utilities"), tier == "developer")

    def test_tiers_and_admin_nickname_flow(self):
        code, license_id = issue_code(
            tier="premium",
            nickname="테스터 A",
        )
        self.assertEqual(license_tier_allows("basic", "level_hunt_movement"), False)
        self.assertTrue(license_tier_allows("premium", "level_hunt_movement"))
        self.assertTrue(license_tier_allows("basic", "ordinary_boss_move"))

        redeem_result = redeem(
            RedeemRequest(code=code, device_id=self.first_device)
        )
        self.assertEqual(redeem_result["tier"], "premium")
        status = get_license_status(license_id)
        self.assertEqual(status["nickname"], "테스터 A")

        set_license_nickname(license_id, "사람 1")
        status = set_license_tier(license_id, "basic")
        self.assertEqual(status["nickname"], "사람 1")
        self.assertEqual(status["tier"], "basic")

        with self.assertRaises(ValueError):
            set_license_nickname(license_id, " " * 2)
        with self.assertRaises(ValueError):
            set_license_tier(license_id, "gold")

    def test_unlimited_code_and_session_last_until_revoked(self):
        code, license_id = issue_code(expires_days=0)
        token = redeem(
            RedeemRequest(code=code, device_id=self.first_device)
        )["access_token"]

        with sqlite3.connect(os.environ["AUTH_DB_PATH"]) as connection:
            code_expiry, used = connection.execute(
                "SELECT expires_at, used FROM auth_codes"
            ).fetchone()
            session_expiry = connection.execute(
                "SELECT expires_at FROM sessions"
            ).fetchone()[0]
        self.assertIsNone(code_expiry)
        self.assertEqual(used, 1)
        self.assertIsNone(session_expiry)
        self.assertEqual(
            verify(
                VerifyRequest(device_id=self.first_device),
                f"Bearer {token}",
            )["license_id"],
            license_id,
        )

        revoke_user(license_id)
        with self.assertRaises(HTTPException):
            verify(
                VerifyRequest(device_id=self.first_device),
                f"Bearer {token}",
            )

    def test_health_check(self):
        self.assertEqual(health_check(), {"ok": True})

    def test_app_update_policy_is_validated_and_returned_by_verify(self):
        policy = set_app_update(
            "1.2.3",
            "0.0.0",
            "https://example.com/KURO_HELPER.zip",
            "a" * 64,
            "안정성 개선",
        )
        self.assertEqual(policy["version"], "1.2.3")
        self.assertEqual(policy["minimum_version"], "0.0.0")
        self.assertEqual(get_app_update(), policy)

        code, _license_id = issue_code()
        token = redeem(
            RedeemRequest(code=code, device_id=self.first_device)
        )["access_token"]
        result = verify(
            VerifyRequest(device_id=self.first_device),
            f"Bearer {token}",
        )
        self.assertEqual(result["app_update"], policy)

        with self.assertRaises(ValueError):
            set_app_update(
                "1.2.3",
                "2.0.0",
                "https://example.com/KURO_HELPER.zip",
                "a" * 64,
            )
        with self.assertRaises(ValueError):
            set_app_update(
                "1.2.3",
                "0.0.0",
                "http://example.com/KURO_HELPER.zip",
                "a" * 64,
            )

        clear_app_update()
        self.assertIsNone(get_app_update())

    def test_app_update_state_distinguishes_optional_and_required(self):
        policy = {
            "version": "1.2.0",
            "minimum_version": "1.1.0",
            "download_url": "https://example.com/KURO_HELPER.zip",
            "sha256": "a" * 64,
        }
        self.assertEqual(update_state(policy, "1.0.9"), "required")
        self.assertEqual(update_state(policy, "1.1.0"), "optional")
        self.assertIsNone(update_state(policy, "1.2.0"))
        with self.assertRaises(UpdateError):
            update_state({**policy, "download_url": "http://example.com/file.zip"})

    def test_partial_update_zip_accepts_assets_and_rejects_user_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            patch_archive = root / "asset-patch.zip"
            with zipfile.ZipFile(patch_archive, "w") as archive:
                archive.writestr(
                    "KURO_HELPER/assets/boss/new-template.png",
                    b"image",
                )
            payload = _extract_package(patch_archive, root / "extracted")
            self.assertEqual(
                (payload / "assets" / "boss" / "new-template.png").read_bytes(),
                b"image",
            )

            settings_archive = root / "settings-patch.zip"
            with zipfile.ZipFile(settings_archive, "w") as archive:
                archive.writestr("KURO_HELPER/settings.json", "{}")
            with self.assertRaisesRegex(UpdateError, "개인 설정"):
                _extract_package(settings_archive, root / "settings-extracted")

    def test_equipment_window_is_clicked_open_only_when_header_is_missing(self):
        cancel_event = Event()
        frame = object()
        with (
            patch("core.equipment_window_controls._move_cursor_to_safe_corner"),
            patch(
                "core.equipment_window_controls._load_template",
                side_effect=[object(), object()],
            ),
            patch(
                "core.equipment_window_controls.capture_window",
                return_value=frame,
            ),
            patch(
                "core.equipment_window_controls._find_template",
                side_effect=[(None, 0.2), ((12, 18), 0.95), ((40, 50), 0.91)],
            ),
            patch(
                "core.equipment_window_controls.win32gui.ClientToScreen",
                return_value=(112, 118),
            ),
            patch("core.equipment_window_controls.click_active_at_screen") as click,
        ):
            opened = ensure_equipment_window_open(
                1,
                "아이템 인벤토리",
                "header.png",
                "inventory.png",
                cancel_event,
                lambda _message: None,
            )
        self.assertTrue(opened)
        click.assert_called_once_with(112, 118)

        with (
            patch("core.equipment_window_controls._move_cursor_to_safe_corner"),
            patch(
                "core.equipment_window_controls._load_template",
                return_value=object(),
            ),
            patch(
                "core.equipment_window_controls.capture_window",
                return_value=frame,
            ),
            patch(
                "core.equipment_window_controls._find_template",
                return_value=((20, 30), 0.95),
            ),
            patch("core.equipment_window_controls.click_active_at_screen") as click,
        ):
            opened = ensure_equipment_window_open(
                1,
                "장비 창",
                "header.png",
                "equipment.png",
                cancel_event,
                lambda _message: None,
            )
        self.assertTrue(opened)
        click.assert_not_called()

    def test_equipment_window_close_toggles_open_window_and_verifies_closed(self):
        with (
            patch(
                "core.equipment_window_controls._load_template",
                side_effect=[object(), object()],
            ),
            patch(
                "core.equipment_window_controls.capture_window",
                return_value=object(),
            ),
            patch(
                "core.equipment_window_controls._find_template",
                side_effect=[((12, 18), 0.95), ((40, 50), 0.95), (None, 0.2)],
            ),
            patch(
                "core.equipment_window_controls.win32gui.ClientToScreen",
                return_value=(140, 150),
            ),
            patch("core.equipment_window_controls.click_active_at_screen") as click,
            patch("core.equipment_window_controls.time.sleep"),
        ):
            closed = close_equipment_window_if_open(
                1,
                "아이템 인벤토리",
                "header.png",
                "inventory.png",
                lambda _message: None,
            )

        self.assertTrue(closed)
        click.assert_called_once_with(140, 150)

    def test_equipment_apply_closes_inventory_after_work(self):
        worker = equipment_apply_worker.EquipmentApplyWorker.__new__(
            equipment_apply_worker.EquipmentApplyWorker
        )
        worker.process_name = "MapleStory.exe"
        worker.update_status = unittest.mock.Mock()
        worker.last_error = None

        with (
            patch("core.equipment_apply_worker.find_hwnd_by_process_name", return_value=1),
            patch("core.equipment_apply_worker.win32gui.IsIconic", return_value=False),
            patch("core.equipment_apply_worker.win32gui.SetForegroundWindow"),
            patch("core.equipment_apply_worker.win32gui.GetForegroundWindow", return_value=1),
            patch("core.equipment_apply_worker.time.sleep"),
            patch("core.equipment_apply_worker.capture_window", return_value=object()),
            patch(
                "core.equipment_apply_worker.ensure_equipment_window_open",
                return_value=(True, (12, 18)),
            ),
            patch("core.equipment_apply_worker._load_template", return_value=object()),
            patch(
                "core.equipment_apply_worker.close_equipment_window_if_open",
                return_value=True,
            ) as close_window,
        ):
            worker._run([], 500, Event())

        self.assertEqual(
            close_window.call_args.args[:4],
            (
                1,
                "아이템 인벤토리",
                equipment_apply_worker.ITEM_INVENTORY_HEADER_TEMPLATE,
                "assets/equipment/Inventory.png",
            ),
        )

    def test_equipment_removal_closes_equipment_window_after_work(self):
        worker = EquipmentRemovalWorker.__new__(EquipmentRemovalWorker)
        worker.process_name = "MapleStory.exe"
        worker.update_status = unittest.mock.Mock()
        worker.last_error = None

        with (
            patch("core.equipment_removal_worker.find_hwnd_by_process_name", return_value=1),
            patch("core.equipment_removal_worker.win32gui.IsIconic", return_value=False),
            patch("core.equipment_removal_worker.win32gui.SetForegroundWindow"),
            patch("core.equipment_removal_worker.win32gui.GetForegroundWindow", return_value=1),
            patch("core.equipment_removal_worker.time.sleep"),
            patch(
                "core.equipment_removal_worker.ensure_equipment_window_open",
                return_value=True,
            ),
            patch(
                "core.equipment_removal_worker.close_equipment_window_if_open",
                return_value=True,
            ) as close_window,
        ):
            worker._run([], 500, Event())

        self.assertEqual(
            close_window.call_args.args[:4],
            (
                1,
                "장비 창",
                "assets/equipment/equipment_inventory_header.png",
                "assets/equipment/Equipment.png",
            ),
        )

    def test_inventory_expands_only_when_plus_template_is_visible(self):
        frame = np.ones((600, 800, 3), dtype=np.uint8)
        cancel_event = Event()
        status_messages = []
        with (
            patch("core.equipment_window_controls._move_cursor_to_safe_corner"),
            patch(
                "core.equipment_window_controls._load_template",
                side_effect=[object(), object()],
            ),
            patch(
                "core.equipment_window_controls.capture_window",
                return_value=frame,
            ),
            patch(
                "core.equipment_window_controls._find_template",
                side_effect=[
                    ((12, 18), 0.95), ((40, 50), 0.95),
                    ((12, 18), 0.95), (None, 0.2),
                ],
            ),
            patch(
                "core.equipment_window_controls.win32gui.ClientToScreen",
                side_effect=[(140, 150), (792, 592)],
            ),
            patch(
                "core.equipment_window_controls.win32gui.GetClientRect",
                return_value=(0, 0, 800, 600),
            ),
            patch("core.equipment_window_controls.click_active_at_screen") as click,
            patch(
                "core.equipment_window_controls.move_cursor_to_screen"
            ) as move_cursor,
        ):
            opened = ensure_equipment_window_open(
                1,
                "아이템 인벤토리",
                "header.png",
                "inventory.png",
                cancel_event,
                status_messages.append,
                expand_button_template="assets/equipment/+.png",
            )

        self.assertTrue(opened)
        click.assert_called_once_with(140, 150)
        move_cursor.assert_called_once_with(792, 592)
        self.assertTrue(any("확장" in message for message in status_messages))

        with (
            patch("core.equipment_window_controls._move_cursor_to_safe_corner"),
            patch(
                "core.equipment_window_controls._load_template",
                side_effect=[object(), object()],
            ),
            patch(
                "core.equipment_window_controls.capture_window",
                return_value=frame,
            ),
            patch(
                "core.equipment_window_controls._find_template",
                side_effect=[((12, 18), 0.95), (None, 0.2)],
            ),
            patch("core.equipment_window_controls.click_active_at_screen") as click,
        ):
            opened = ensure_equipment_window_open(
                1,
                "아이템 인벤토리",
                "header.png",
                "inventory.png",
                cancel_event,
                lambda _message: None,
                expand_button_template="assets/equipment/+.png",
            )

        self.assertTrue(opened)
        click.assert_not_called()

    def test_inventory_window_returns_header_anchor_before_expanding(self):
        frame = np.ones((600, 800, 3), dtype=np.uint8)
        with (
            patch("core.equipment_window_controls._move_cursor_to_safe_corner"),
            patch(
                "core.equipment_window_controls._load_template",
                side_effect=[object(), object()],
            ),
            patch(
                "core.equipment_window_controls.capture_window",
                return_value=frame,
            ),
            patch(
                "core.equipment_window_controls._find_template",
                side_effect=[
                    ((12, 18), 0.95), ((40, 50), 0.95),
                    ((12, 18), 0.95), (None, 0.2),
                ],
            ),
            patch(
                "core.equipment_window_controls.win32gui.ClientToScreen",
                side_effect=[(140, 150), (792, 592)],
            ),
            patch(
                "core.equipment_window_controls.win32gui.GetClientRect",
                return_value=(0, 0, 800, 600),
            ),
            patch("core.equipment_window_controls.click_active_at_screen"),
            patch(
                "core.equipment_window_controls.move_cursor_to_screen"
            ) as move_cursor,
        ):
            result = ensure_equipment_window_open(
                1,
                "아이템 인벤토리",
                "header.png",
                "inventory.png",
                Event(),
                lambda _message: None,
                expand_button_template="assets/equipment/+.png",
                return_header_point=True,
            )

        self.assertEqual(result, (True, (12, 18)))
        move_cursor.assert_called_once_with(792, 592)

    def test_expanded_inventory_reuses_header_anchor_when_label_is_hidden(self):
        with patch.object(
            equipment_apply_worker,
            "_find_scaled_template",
            return_value=(None, 0.41),
        ):
            point, score = equipment_apply_worker._find_inventory_header_point(
                object(),
                object(),
                fallback_point=(12, 18),
            )

        self.assertEqual(point, (12, 18))
        self.assertEqual(score, 0.41)

    def test_equipment_inventory_templates_match_above_1_4_scale(self):
        template = np.random.default_rng(7).integers(
            0,
            256,
            size=(12, 32, 3),
            dtype=np.uint8,
        )
        resized = cv2.resize(template, (58, 22), interpolation=cv2.INTER_LINEAR)
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        frame[24:46, 30:88] = resized

        window_point, window_score = equipment_window_controls._find_template(
            frame,
            template,
        )
        apply_point, apply_score = equipment_apply_worker._find_scaled_template(
            frame,
            template,
        )

        self.assertEqual(window_point, (59, 35))
        self.assertGreaterEqual(window_score, 0.99)
        self.assertEqual(apply_point, (59, 35))
        self.assertGreaterEqual(apply_score, 0.99)

    def test_multi_boss_plan_uses_all_equipment_management_storage_counts(self):
        class Entry:
            def __init__(self, value):
                self.value = value

            def get(self):
                return str(self.value)

        app = SimpleNamespace(
            saved_settings={
                "equipment_apply_presets": [{}],
                "equipment_removal_presets": [{}],
                "storage_presets": [{
                    "storage_equipment_count": 3,
                    "storage_installation_count": 4,
                    "storage_cash_count": 2,
                    "storage_command_key": "-",
                    "storage_delay_ms": 300,
                }],
            },
            equipment_area=object(),
            equipment_apply_inventory_area=object(),
            equipment_apply_equipment_count=Entry(2),
            equipment_apply_cash_count=Entry(1),
            storage_count_entries={
                "storage_equipment_count": Entry(3),
                "storage_installation_count": Entry(4),
                "storage_cash_count": Entry(2),
            },
            equipment_apply_delay_entry=Entry(100),
            equipment_removal_delay_entry=Entry(200),
            storage_delay_entry=Entry(300),
            storage_command_key=Entry("-"),
            equipment_slot_states={"무기": 1},
            dismantle_categories={
                category: SimpleNamespace(get=lambda: 1)
                for category in main_window.DECOMPOSITION_CATEGORIES.values()
            },
            dismantle_interaction_key=Entry("space"),
            home_travel_toggle_hangul=Entry(0),
            _get_equipment_apply_slot_client_point=lambda _slot, _size: (10, 20),
            _get_equipment_slot_client_point=lambda _slot, _size: (30, 40),
        )
        with (
            patch("ui.main_window.find_hwnd_by_process_name", return_value=1),
            patch(
                "ui.main_window.capture_window",
                return_value=SimpleNamespace(shape=(100, 200, 3)),
            ),
        ):
            plan = KuroHelper._build_multi_boss_equipment_plan(
                app, {"withdraw", "deposit"},
            )

        self.assertEqual(
            plan["storage_counts"],
            {"equipment": 3, "installation": 4, "cash": 2},
        )
        for stage in ("withdraw", "deposit"):
            self.assertEqual(plan["storage_plans"][stage]["counts"], plan["storage_counts"])

    def test_multi_boss_refreshes_equipment_coordinates_from_current_client_size(self):
        app = SimpleNamespace(
            _get_equipment_apply_slot_client_point=(
                unittest.mock.Mock(side_effect=lambda slot_number, client_size, **kwargs: ("inventory", slot_number, client_size))
            ),
            _get_equipment_slot_client_point=(
                unittest.mock.Mock(side_effect=lambda slot_name, client_size, **kwargs: ("equipment", slot_name, client_size))
            ),
        )
        plan = {
            "equipment_apply_inventory_area": (10, 20, 200, 300),
            "equipment_area": (300, 20, 200, 300),
            "apply_batches": [("equipment", [(1, (10, 20)), (2, (30, 40))])],
            "removal_actions": [("무기", (50, 60), 2)],
        }
        frame = SimpleNamespace(shape=(720, 1280, 3))

        with (
            patch("ui.main_window.find_hwnd_by_process_name", return_value=7),
            patch("ui.main_window.capture_window", return_value=frame),
        ):
            KuroHelper._refresh_multi_boss_equipment_plan_coordinates(app, plan)

        self.assertEqual(
            plan["apply_batches"],
            [("equipment", [(1, ("inventory", 1, (1280, 720))), (2, ("inventory", 2, (1280, 720)))])],
        )
        self.assertEqual(
            plan["removal_actions"],
            [("무기", ("equipment", "무기", (1280, 720)), 2)],
        )
        self.assertEqual(
            [call.kwargs["area"] for call in app._get_equipment_apply_slot_client_point.call_args_list],
            [plan["equipment_apply_inventory_area"]] * 2,
        )
        app._get_equipment_slot_client_point.assert_called_once_with(
            "무기", (1280, 720), area=plan["equipment_area"],
        )

    def test_storage_uses_fixed_menu_delay_and_configured_item_delay(self):
        worker = StorageWorker(unittest.mock.Mock())
        worker._tap_key = unittest.mock.Mock()
        for action, selection_keys in (
            ("넣기", ["-", "space", "space"]),
            ("빼기", ["-", "space", "down", "space"]),
        ):
            worker._tap_key.reset_mock()
            with (
                patch("core.storage_worker.find_hwnd_by_process_name", return_value=1),
                patch("core.storage_worker.win32gui.IsIconic", return_value=False),
                patch("core.storage_worker.win32gui.SetForegroundWindow"),
                patch("core.storage_worker.win32gui.GetForegroundWindow", return_value=1),
                patch("core.storage_worker.time.sleep"),
                patch("core.storage_worker._load_template", return_value=object()),
                patch(
                    "core.storage_worker.ensure_equipment_window_open",
                    return_value=(True, (12, 18)),
                ),
                patch("core.storage_worker.close_equipment_window_if_open", return_value=False),
                patch.object(worker, "_wait_for_storage_dialog", return_value=True) as wait_dialog,
                patch.object(worker, "_dismiss_escape_menu", return_value=False),
                patch.object(
                    worker, "_count_inventory_slots",
                    side_effect=(1, 0) if action == "넣기" else (0, 1),
                ) as count_slots,
            ):
                worker._run(
                    build_storage_batches(
                        action,
                        "-",
                        {"equipment": 1, "installation": 0, "cash": 0},
                    ),
                    0.12,
                    Event(),
                    action=action,
                )

            expected_keys = selection_keys + ["space", "space", "esc"]
            delays = [call.args[2] for call in worker._tap_key.call_args_list]
            self.assertEqual(
                delays,
                [0] + [STORAGE_MENU_DELAY_SECONDS] * (len(selection_keys) - 1)
                + [0.12, 0.12, STORAGE_MENU_DELAY_SECONDS],
            )
            self.assertEqual(
                [call.args[1] for call in worker._tap_key.call_args_list],
                expected_keys,
            )
            wait_dialog.assert_called_once()
            self.assertEqual(count_slots.call_count, 2)
            self.assertFalse(count_slots.call_args.kwargs["ensure_tab"])
            self.assertIsNone(worker.last_error)
            worker.update_status.assert_called_with(
                f"창고 작업 완료 (1개 분류, 확인된 {action} 1칸)"
            )

    def test_maple_window_moves_to_top_left_without_resizing(self):
        app = KuroHelper.__new__(KuroHelper)
        app.update_status_text = unittest.mock.Mock()

        with (
            patch(
                "ui.main_window.find_hwnd_by_process_name",
                return_value=55,
            ) as find_window,
            patch("ui.main_window.win32gui.IsIconic", return_value=False),
            patch(
                "ui.main_window.win32gui.GetWindowPlacement",
                return_value=(
                    0,
                    main_window.win32con.SW_SHOWMAXIMIZED,
                    (0, 0),
                    (0, 0),
                    (0, 0, 0, 0),
                ),
            ),
            patch("ui.main_window.win32gui.ShowWindow") as show_window,
            patch("ui.main_window.win32gui.GetWindowText", return_value="아마테라스"),
            patch(
                "ui.main_window.win32gui.GetWindowRect",
                side_effect=[(100, 100, 900, 700), (0, 0, 800, 600)],
            ),
            patch("ui.main_window.win32gui.SetWindowPos") as set_window_pos,
        ):
            app.move_game_window_to_top_left()

        show_window.assert_called_once_with(55, main_window.win32con.SW_RESTORE)
        set_window_pos.assert_called_once_with(
            55,
            None,
            0,
            0,
            0,
            0,
            main_window.win32con.SWP_NOSIZE
            | main_window.win32con.SWP_NOZORDER
            | main_window.win32con.SWP_NOACTIVATE,
        )
        app.update_status_text.assert_called_once_with(
            "창 이동 완료: 아마테라스 (HWND=55, 위치 0, 0)"
        )
        find_window.assert_called_once_with(
            "MapleStory.exe",
            window_title="아마테라스",
        )

    def test_window_lookup_filters_by_process_and_title(self):
        process = SimpleNamespace(info={"name": "MapleStory.exe", "pid": 22})

        def enumerate_windows(callback, extra):
            callback(101, extra)
            callback(202, extra)

        with (
            patch.object(win_control.psutil, "process_iter", return_value=[process]),
            patch.object(win_control.win32gui, "IsWindowVisible", return_value=True),
            patch.object(
                win_control.win32process,
                "GetWindowThreadProcessId",
                return_value=(1, 22),
            ),
            patch.object(
                win_control.win32gui,
                "GetWindowText",
                side_effect=["MapleStory Launcher", "아마테라스"],
            ) as get_window_text,
            patch.object(
                win_control.win32gui,
                "EnumWindows",
                side_effect=enumerate_windows,
            ),
        ):
            hwnd = win_control.find_hwnd_by_process_name(
                "MapleStory.exe",
                window_title="아마테라스",
            )
            self.assertEqual(hwnd, 202)
            get_window_text.side_effect = ["MapleStory Launcher", "Other title"]
            fallback_hwnd = win_control.find_hwnd_by_process_name(
                "MapleStory.exe",
                window_title="아마테라스",
            )
        self.assertEqual(fallback_hwnd, 101)

    def test_maple_window_move_shows_error_when_no_window_is_found(self):
        app = KuroHelper.__new__(KuroHelper)
        app.update_status_text = unittest.mock.Mock()

        with (
            patch("ui.main_window.find_hwnd_by_process_name", return_value=None),
            patch("ui.main_window.messagebox.showerror") as show_error,
        ):
            app.move_game_window_to_top_left()

        app.update_status_text.assert_called_once_with(
            "창 이동 실패: MapleStory.exe 프로세스의 표시 중인 창을 찾지 못했습니다."
        )
        show_error.assert_called_once()

    def test_boss_rotation_continues_without_sending_resolution_command(self):
        with patch("core.multi_character_tester.Controller"):
            tester = MultiCharacterTester(unittest.mock.Mock())
        tester._validate_templates = unittest.mock.Mock(return_value={})
        tester._require_game_window = unittest.mock.Mock(return_value=7)
        tester._login_character = unittest.mock.Mock(return_value=7)
        tester._wait_for_button = unittest.mock.Mock(return_value=(7, None))
        tester._dismiss_chloe_if_present = unittest.mock.Mock()
        tester._set_game_resolution = unittest.mock.Mock()
        tester._notify_equipment_ownership = unittest.mock.Mock()
        tester.boss_runner = unittest.mock.Mock(return_value="성공")
        cancel_event = Event()

        with patch(
            "core.multi_character_tester.move_window_to_top_left",
            return_value=("game", (0, 0, 800, 600), (0, 0, 800, 600)),
        ):
            tester._run_boss_rotation(
                ["A1", "A2"], "CH.1", "password", ["boss"], "F10", False,
                {}, cancel_event, enabled_stages={"character_cycle", "boss_tour"},
            )

        tester._login_character.assert_called_once_with(
            7, "A2", {}, "CH.1", "password", cancel_event,
        )
        self.assertEqual(tester.boss_runner.call_count, 2)
        self.assertEqual(
            [call.kwargs["character_id"] for call in tester.boss_runner.call_args_list],
            ["A1", "A2"],
        )
        tester._set_game_resolution.assert_not_called()
        tester._keyboard.type.assert_not_called()
        tester.update_status.assert_called_with("다캐릭 보스돌이 완료")

    def test_character_login_returns_without_sending_resolution_command(self):
        with patch("core.multi_character_tester.Controller"):
            tester = MultiCharacterTester(unittest.mock.Mock())
        for name in ("_return_to_login", "_login", "_select_server_and_channel", "_select_character"):
            setattr(tester, name, unittest.mock.Mock(return_value=7))
        tester._click_button = unittest.mock.Mock()
        tester._wait_for_button = unittest.mock.Mock(return_value=(7, None))
        tester._dismiss_chloe_if_present = unittest.mock.Mock()
        tester._set_game_resolution = unittest.mock.Mock()
        cancel_event = Event()

        self.assertEqual(
            tester._login_character(7, "A2", {}, "CH.1", "password", cancel_event),
            7,
        )
        tester._click_button.assert_called_once_with(
            7, "게임 시작", cancel_event, timeout=30,
        )
        tester._wait_for_button.assert_called_once_with(
            7, "게임 메뉴", cancel_event, timeout=120,
        )
        tester._dismiss_chloe_if_present.assert_called_once_with(7, cancel_event)
        tester._set_game_resolution.assert_not_called()
        tester._keyboard.type.assert_not_called()
        tester._keyboard.press.assert_not_called()

    def test_admin_check_reads_member_permissions_and_configured_role(self):
        administrator = SimpleNamespace(
            user=SimpleNamespace(
                guild_permissions=SimpleNamespace(administrator=True),
                roles=[],
            )
        )
        role_admin = SimpleNamespace(
            user=SimpleNamespace(
                guild_permissions=SimpleNamespace(administrator=False),
                roles=[SimpleNamespace(id=42)],
            )
        )
        non_admin = SimpleNamespace(
            user=SimpleNamespace(roles=[])
        )

        self.assertTrue(_is_admin(administrator, 42))
        self.assertTrue(_is_admin(role_admin, 42))
        self.assertFalse(_is_admin(non_admin, 42))

    def test_discord_commands_include_code_retrieval_and_update_management(self):
        with patch.dict(
            os.environ,
            {
                "DISCORD_BOT_TOKEN": "test-token",
                "DISCORD_GUILD_ID": "123456789012345678",
                "DISCORD_ADMIN_ROLE_ID": "234567890123456789",
            },
        ):
            bot, _token = create_bot()
        commands = {
            command.name
            for command in bot.tree.get_commands(
                guild=discord.Object(id=bot.guild_id)
            )
        }
        self.assertIn("getcode", commands)
        self.assertIn("setlicenseexpiry", commands)
        self.assertIn("setappupdate", commands)
        self.assertIn("clearappupdate", commands)
        reset_command = next(
            command for command in bot.tree.get_commands(
                guild=discord.Object(id=bot.guild_id)
            ) if command.name == "resetdevice"
        )
        expires_days = next(
            parameter for parameter in reset_command.parameters
            if parameter.name == "expires_days"
        )
        self.assertTrue(expires_days.required)
        for name in ("issuecode", "settier"):
            command = next(command for command in bot.tree.get_commands(
                guild=discord.Object(id=bot.guild_id)
            ) if command.name == name)
            tier = next(parameter for parameter in command.parameters if parameter.name == "tier")
            self.assertEqual({choice.value for choice in tier.choices}, {"basic", "premium", "developer"})

    def test_license_monitor_notifies_when_server_revokes_session(self):
        revoked = Event()
        monitor = LicenseMonitor(revoked.set, interval_seconds=0.01)
        with (
            patch("core.auth_client._load_api_url", return_value="https://auth.test"),
            patch("core.auth_client._get_machine_id", return_value=self.first_device),
            patch("core.auth_client._read_saved_token", return_value="saved-token"),
            patch(
                "core.auth_client._request",
                side_effect=AuthenticationError("revoked", status_code=401),
            ),
        ):
            monitor.start()
            try:
                self.assertTrue(revoked.wait(timeout=1))
            finally:
                monitor.stop()

    def test_license_monitor_notifies_when_tier_changes(self):
        tier_changed = Event()
        observed_tiers = []

        def on_tier_changed(tier):
            observed_tiers.append(tier)
            tier_changed.set()

        monitor = LicenseMonitor(
            lambda: None,
            interval_seconds=0.01,
            on_tier_changed=on_tier_changed,
            initial_tier="basic",
        )
        with (
            patch("core.auth_client._load_api_url", return_value="https://auth.test"),
            patch("core.auth_client._get_machine_id", return_value=self.first_device),
            patch("core.auth_client._read_saved_token", return_value="saved-token"),
            patch(
                "core.auth_client._request",
                return_value={"authorized": True, "tier": "premium"},
            ),
        ):
            monitor.start()
            try:
                self.assertTrue(tier_changed.wait(timeout=1))
                self.assertEqual(observed_tiers, ["premium"])
            finally:
                monitor.stop()

    def test_license_monitor_notifies_when_update_becomes_required(self):
        update_required = Event()
        observed_versions = []
        version_parts = [int(part) for part in APP_VERSION.split(".")]
        version_parts[-1] += 1
        required_version = ".".join(str(part) for part in version_parts)

        def on_forced_update(version):
            observed_versions.append(version)
            update_required.set()

        monitor = LicenseMonitor(
            lambda: None,
            interval_seconds=0.01,
            on_forced_update=on_forced_update,
        )
        policy = {
            "version": required_version,
            "minimum_version": required_version,
            "download_url": "https://example.com/KURO_HELPER.zip",
            "sha256": "a" * 64,
        }
        with (
            patch("core.auth_client._load_api_url", return_value="https://auth.test"),
            patch("core.auth_client._get_machine_id", return_value=self.first_device),
            patch("core.auth_client._read_saved_token", return_value="saved-token"),
            patch(
                "core.auth_client._request",
                return_value={
                    "authorized": True,
                    "tier": "basic",
                    "app_update": policy,
                },
            ),
        ):
            monitor.start()
            try:
                self.assertTrue(update_required.wait(timeout=1))
                self.assertEqual(observed_versions, [required_version])
            finally:
                monitor.stop()


if __name__ == "__main__":
    unittest.main()
