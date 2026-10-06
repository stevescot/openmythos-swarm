#!/usr/bin/env python3
"""Sensors for the E.ON Next integration.

Includes the standard meter/tariff sensors plus the "cheap window" sensors that
read the real half-hourly rate schedule from the new E.ON interface so automations
can tell whether *now* is inside a cheap period (and see the whole day's windows)
instead of relying on a hardcoded clock heuristic.
"""

import logging

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
)
from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.const import UnitOfEnergy, UnitOfVolume
from homeassistant.util import dt as dt_util

from . import DOMAIN
from .eonnext import (
    DEFAULT_CHEAP_THRESHOLD_P,
    METER_TYPE_ELECTRIC,
    METER_TYPE_GAS,
)

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass, config_entry, async_add_entities):
    """Set up sensors from a config entry."""

    api = hass.data[DOMAIN][config_entry.entry_id]

    entities = []
    for account in api.accounts:
        for meter in account.meters:
            if await meter.has_reading():
                entities.append(LatestReadingDateSensor(meter))
                if meter.get_type() == METER_TYPE_ELECTRIC:
                    entities.append(LatestElectricKwhSensor(meter))
                if meter.get_type() == METER_TYPE_GAS:
                    entities.append(LatestGasCubicMetersSensor(meter))
                    entities.append(LatestGasKwhSensor(meter))

        for charger in account.ev_chargers:
            entities.append(SmartChargingScheduleSensor(charger))
            entities.append(NextChargeStartSensor(charger))
            entities.append(NextChargeEndSensor(charger))

        # Tariff sensors for the account
        if account.tariff_data:
            entities.append(TariffNameSensor(account))
            entities.append(StandingChargeSensor(account))
            entities.append(UnitRateSensor(account))
            # New: real cheap-window sensors from the E.ON-assigned charging
            # schedule (authoritative) with the half-hourly rate schedule as
            # fallback.
            entities.append(CheapWindowActiveSensor(account))
            entities.append(CheapWindowsTodaySensor(account))
            entities.append(NextCheapWindowSensor(account))
            entities.append(AssignedChargingScheduleSensor(account))

        if account.saving_sessions:
            entities.append(SavingSessionsSensor(account))

        entities.append(BillingHistorySensor(account))

    async_add_entities(entities, update_before_add=True)


def _active_agreements(account):
    """Return agreements still valid at `now`."""
    now = dt_util.now()
    return [
        a for a in (account.tariff_data or [])
        if not a.get("validTo") or dt_util.parse_datetime(a["validTo"]) > now
    ]


class LatestReadingDateSensor(SensorEntity):
    """Date of latest meter reading."""

    def __init__(self, meter):
        self.meter = meter
        self._attr_name = self.meter.get_serial() + " Reading Date"
        self._attr_device_class = SensorDeviceClass.DATE
        self._attr_icon = "mdi:calendar"
        self._attr_unique_id = self.meter.get_serial() + "__reading_date"

    async def async_update(self) -> None:
        self._attr_native_value = await self.meter.get_latest_reading_date()


class LatestElectricKwhSensor(SensorEntity):
    """Latest electricity meter reading."""

    def __init__(self, meter):
        self.meter = meter
        self._attr_name = self.meter.get_serial() + " Electricity"
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = "total"
        self._attr_icon = "mdi:meter-electric-outline"
        self._attr_unique_id = self.meter.get_serial() + "__electricity_kwh"

    async def async_update(self) -> None:
        self._attr_native_value = await self.meter.get_latest_reading()


class LatestGasKwhSensor(SensorEntity):
    """Latest gas meter reading in kWh."""

    def __init__(self, meter):
        self.meter = meter
        self._attr_name = self.meter.get_serial() + " Gas kWh"
        self._attr_device_class = SensorDeviceClass.ENERGY
        self._attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
        self._attr_state_class = "total"
        self._attr_icon = "mdi:meter-gas-outline"
        self._attr_unique_id = self.meter.get_serial() + "__gas_kwh"

    async def async_update(self) -> None:
        self._attr_native_value = await self.meter.get_latest_reading_kwh()


class LatestGasCubicMetersSensor(SensorEntity):
    """Latest gas meter reading in cubic meters."""

    def __init__(self, meter):
        self.meter = meter
        self._attr_name = self.meter.get_serial() + " Gas"
        self._attr_device_class = SensorDeviceClass.GAS
        self._attr_native_unit_of_measurement = UnitOfVolume.CUBIC_METERS
        self._attr_state_class = "total"
        self._attr_icon = "mdi:meter-gas-outline"
        self._attr_unique_id = self.meter.get_serial() + "__gas_m3"

    async def async_update(self) -> None:
        self._attr_native_value = await self.meter.get_latest_reading()


class SmartChargingScheduleSensor(SensorEntity):
    """Smart charging schedule."""

    def __init__(self, charger):
        self.charger = charger
        self._attr_name = self.charger.get_serial() + " Smart Charging Schedule"
        self._attr_icon = "mdi:ev-station"
        self._attr_unique_id = self.charger.get_serial() + "__smart_charging_schedule"
        self._attr_extra_state_attributes = {}

    async def async_update(self) -> None:
        schedule = await self.charger.get_schedule()
        if schedule:
            self._attr_native_value = "Active"
            self._attr_extra_state_attributes["schedule"] = schedule
        elif schedule is not None:
            self._attr_native_value = "No Schedule"
            self._attr_extra_state_attributes["schedule"] = []
        else:
            self._attr_native_value = "Unknown"


class NextChargeStartSensor(SensorEntity):
    """Start time of next charge."""

    def __init__(self, charger):
        self.charger = charger
        self._attr_name = self.charger.get_serial() + " Next Charge Start"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-start"
        self._attr_unique_id = self.charger.get_serial() + "__next_charge_start"

    async def async_update(self) -> None:
        schedule = await self.charger.get_schedule()
        if schedule and len(schedule) > 0:
            self._attr_native_value = dt_util.parse_datetime(schedule[0]["start"])
        else:
            self._attr_native_value = None


class NextChargeEndSensor(SensorEntity):
    """End time of next charge."""

    def __init__(self, charger):
        self.charger = charger
        self._attr_name = self.charger.get_serial() + " Next Charge End"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-end"
        self._attr_unique_id = self.charger.get_serial() + "__next_charge_end"

    async def async_update(self) -> None:
        schedule = await self.charger.get_schedule()
        if schedule and len(schedule) > 0:
            self._attr_native_value = dt_util.parse_datetime(schedule[0]["end"])
        else:
            self._attr_native_value = None


class TariffNameSensor(SensorEntity):
    """Active tariff name for the account."""

    def __init__(self, account):
        self.account = account
        self._attr_name = "Account Tariff Name"
        self._attr_icon = "mdi:file-document-outline"
        self._attr_unique_id = f"{self.account.account_number}__tariff_name"

    async def async_update(self) -> None:
        await self.account._load_tariff_data()
        active = _active_agreements(self.account)
        if active:
            tariff = active[0].get("tariff", {})
            self._attr_native_value = tariff.get("displayName") or tariff.get("fullName")
            self._attr_extra_state_attributes = {
                "tariff_code": tariff.get("tariffCode"),
                "valid_from": active[0].get("validFrom"),
                "valid_to": active[0].get("validTo"),
            }
        else:
            self._attr_native_value = None


class StandingChargeSensor(SensorEntity):
    """Daily standing charge for the account."""

    def __init__(self, account):
        self.account = account
        self._attr_name = "Account Standing Charge"
        self._attr_icon = "mdi:currency-gbp"
        self._attr_unit_of_measurement = "GBP/day"
        self._attr_unique_id = f"{self.account.account_number}__standing_charge"

    async def async_update(self) -> None:
        await self.account._load_tariff_data()
        active = _active_agreements(self.account)
        if active:
            tariff = active[0].get("tariff", {})
            standing_charge = tariff.get("standingCharge")
            self._attr_native_value = (
                round(standing_charge / 100, 4) if standing_charge is not None else None
            )
        else:
            self._attr_native_value = None


class UnitRateSensor(SensorEntity):
    """Unit rate for the account.

    Handles both the wide-range shape (a cheap block + a peak block) and the
    48-slot half-hourly shape that E.ON Next returns.
    """

    def __init__(self, account):
        self.account = account
        self._attr_name = "Account Unit Rate"
        self._attr_icon = "mdi:currency-gbp"
        self._attr_unit_of_measurement = "GBP/kWh"
        self._attr_unique_id = f"{self.account.account_number}__unit_rate"
        self._attr_extra_state_attributes = {}

    async def async_update(self) -> None:
        await self.account._load_tariff_data()
        active = _active_agreements(self.account)
        if not active:
            self._attr_native_value = None
            return

        tariff = active[0].get("tariff", {})
        unit_rate = tariff.get("unitRate")

        if unit_rate is None and self.account.rate_schedule:
            rate = self.account.current_rate_p()
            if rate is not None:
                unit_rate = rate
            cheap = self.account.is_cheap_now(DEFAULT_CHEAP_THRESHOLD_P)
            self._attr_extra_state_attributes = {
                "meter_point": active[0].get("meterPoint", {}).get("mpan"),
                "current_period": "Off-Peak" if cheap else "Peak",
                "rate_slots": len(self.account.rate_schedule),
            }

        if unit_rate is not None:
            self._attr_native_value = round(unit_rate / 100, 4)
            if not self._attr_extra_state_attributes:
                self._attr_extra_state_attributes = {
                    "meter_point": active[0].get("meterPoint", {}).get("mpan")
                }
        else:
            self._attr_native_value = None


class CheapWindowActiveSensor(BinarySensorEntity):
    """Binary sensor: is the current half-hour slot cheap?

    Reads the real half-hourly rate schedule from the new E.ON interface rather
    than a clock heuristic, so a whole-day cheap period is reported correctly.
    """

    def __init__(self, account, threshold_p=DEFAULT_CHEAP_THRESHOLD_P):
        self.account = account
        self.threshold_p = threshold_p
        self._attr_name = "Cheap Electricity Window Active"
        self._attr_icon = "mdi:lightning-bolt"
        self._attr_unique_id = f"{self.account.account_number}__cheap_window_active"

    async def async_update(self) -> None:
        await self.account._load_tariff_data()
        for charger in self.account.ev_chargers:
            await charger.update()

        # Authoritative source: the charging windows E.ON assigned to the car.
        charging = self.account.is_charging_window_now()
        windows = self.account.get_charging_windows()
        now = dt_util.now()
        current = next(
            (
                w
                for w in windows
                if dt_util.parse_datetime(w["start"]) <= now < dt_util.parse_datetime(w["end"])
            ),
            None,
        )

        if charging is not None:
            cheap = charging
            source = "eon_assigned_charging_schedule"
        else:
            # Fallback: derive from the half-hourly unit-rate schedule.
            cheap = self.account.is_cheap_now(self.threshold_p)
            source = "eon_half_hourly_rate_schedule"
            rate_windows = self.account.get_cheap_windows(self.threshold_p)
            current = next(
                (
                    w
                    for w in rate_windows
                    if dt_util.parse_datetime(w["start"]) <= now < dt_util.parse_datetime(w["end"])
                ),
                None,
            )

        if cheap is None:
            self._attr_available = False
            self._attr_is_on = None
        else:
            self._attr_available = True
            self._attr_is_on = bool(cheap)
        self._attr_extra_state_attributes = {
            "current_rate_p": self.account.current_rate_p(),
            "threshold_p": self.threshold_p,
            "window_start": current["start"] if current else None,
            "window_end": current["end"] if current else None,
            "source": source,
        }


class CheapWindowsTodaySensor(SensorEntity):
    """All cheap windows for today (from the half-hourly schedule)."""

    def __init__(self, account, threshold_p=DEFAULT_CHEAP_THRESHOLD_P):
        self.account = account
        self.threshold_p = threshold_p
        self._attr_name = "Cheap Windows Today"
        self._attr_icon = "mdi:clock-time-four-outline"
        self._attr_unique_id = f"{self.account.account_number}__cheap_windows_today"

    async def async_update(self) -> None:
        await self.account._load_tariff_data()
        windows = self.account.get_cheap_windows(self.threshold_p)
        today = dt_util.now().date()
        todays = []
        for w in windows:
            start = dt_util.parse_datetime(w["start"])
            if start is None:
                continue
            local = start.astimezone() if start.tzinfo else start
            if local.date() == today:
                todays.append(w)
        self._attr_native_value = len(todays)
        self._attr_extra_state_attributes = {
            "windows": todays,
            "threshold_p": self.threshold_p,
        }


class NextCheapWindowSensor(SensorEntity):
    """Start of the next cheap window (timestamp)."""

    def __init__(self, account, threshold_p=DEFAULT_CHEAP_THRESHOLD_P):
        self.account = account
        self.threshold_p = threshold_p
        self._attr_name = "Next Cheap Window Start"
        self._attr_device_class = SensorDeviceClass.TIMESTAMP
        self._attr_icon = "mdi:clock-start"
        self._attr_unique_id = f"{self.account.account_number}__next_cheap_window"

    async def async_update(self) -> None:
        await self.account._load_tariff_data()
        now = dt_util.now()
        windows = self.account.get_cheap_windows(self.threshold_p)
        for w in windows:
            start = dt_util.parse_datetime(w["start"])
            if start is not None and start > now:
                self._attr_native_value = start
                self._attr_extra_state_attributes = {"rate_p": w["rate_p"]}
                return
        self._attr_native_value = None


class AssignedChargingScheduleSensor(SensorEntity):
    """The EV charging windows E.ON has assigned to this account.

    The state is the number of windows today; attributes carry the full merged
    schedule so you can confirm the integration retrieved the correct charging
    schedule (e.g. an all-day window).
    """

    def __init__(self, account):
        self.account = account
        self._attr_name = "Assigned Charging Schedule"
        self._attr_icon = "mdi:ev-station"
        self._attr_unique_id = f"{self.account.account_number}__assigned_charging_schedule"

    async def async_update(self) -> None:
        for charger in self.account.ev_chargers:
            await charger.update()
        windows = self.account.get_charging_windows()
        today = dt_util.now().date()
        todays = []
        for w in windows:
            start = dt_util.parse_datetime(w["start"])
            if start is None:
                continue
            local = start.astimezone() if start.tzinfo else start
            if local.date() == today:
                todays.append(w)
        self._attr_native_value = len(todays)
        self._attr_extra_state_attributes = {
            "windows_today": todays,
            "windows_all": windows,
            "chargers": [c.get_serial() for c in self.account.ev_chargers],
        }


class SavingSessionsSensor(SensorEntity):
    """Upcoming and active saving sessions."""

    def __init__(self, account):
        self.account = account
        self._attr_name = "Account Saving Sessions"
        self._attr_icon = "mdi:piggy-bank-outline"
        self._attr_unique_id = f"{self.account.account_number}__saving_sessions"

    async def async_update(self) -> None:
        await self.account._load_saving_sessions()
        self._attr_native_value = len(self.account.saving_sessions)
        self._attr_extra_state_attributes = {"sessions": self.account.saving_sessions}


class BillingHistorySensor(SensorEntity):
    """Recent billing history."""

    def __init__(self, account):
        self.account = account
        self._attr_name = "Account Billing History"
        self._attr_icon = "mdi:file-document-edit-outline"
        self._attr_unique_id = f"{self.account.account_number}__billing_history"

    async def async_update(self) -> None:
        await self.account._load_billing_data()
        self._attr_native_value = len(self.account.billing_data)
        self._attr_extra_state_attributes = {"bills": self.account.billing_data}
