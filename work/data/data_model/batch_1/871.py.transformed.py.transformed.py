
from ctypes import (HRESULT, POINTER, Structure, Union, c_float, c_longlong,
                    c_uint32)
from ctypes.wintypes import (BOOL, DWORD, INT, LONG, LPCWSTR, LPWSTR, UINT,
                             ULARGE_INTEGER, VARIANT_BOOL, WORD)
from enum import Enum
import comtypes
import psutil
from comtypes import COMMETHOD, GUID, IUnknown
from comtypes.automation import VARTYPE, VT_BOOL, VT_CLSID, VT_LPWSTR, VT_UI4
from future.utils import python_2_unicode_compatible
IID_Empty = GUID(
    '{00000000-0000-0000-0000-000000000000}')
CLSID_MMDeviceEnumerator = GUID(
    '{BCDE0395-E52F-467C-8E3D-C4579291692E}')
UINT32 = c_uint32
REFERENCE_TIME = c_longlong
class PROPVARIANT_UNION(Union):
        _fields_ = [
            ('lVal', LONG),
            ('uhVal', ULARGE_INTEGER),
            ('boolVal', VARIANT_BOOL),
            ('pwszVal', LPWSTR),
            ('puuid', GUID),
        ]
class PROPVARIANT(Structure):
    _fields_ = [
        ('vt', VARTYPE),
        ('reserved1', WORD),
        ('reserved2', WORD),
        ('reserved3', WORD),
        ('union', PROPVARIANT_UNION),
    ]
    def GetValue(self):
        vt = self.vt
        if vt == VT_BOOL:
            return self.union.boolVal != 0
        elif vt == VT_LPWSTR:
            return self.union.pwszVal
        elif vt == VT_UI4:
            return self.union.lVal
        elif vt == VT_CLSID:
            return
        else:
            return "%s:?" % (vt)
class WAVEFORMATEX(Structure):
    _fields_ = [
        ('wFormatTag', WORD),
        ('nChannels', WORD),
        ('nSamplesPerSec', WORD),
        ('nAvgBytesPerSec', WORD),
        ('nBlockAlign', WORD),
        ('wBitsPerSample', WORD),
        ('cbSize', WORD),
    ]
class ERole(Enum):
    eConsole = 0
    eMultimedia = 1
    eCommunications = 2
    ERole_enum_count = 3
class EDataFlow(Enum):
    eRender = 0
    eCapture = 1
    eAll = 2
    EDataFlow_enum_count = 3
class DEVICE_STATE(Enum):
    ACTIVE = 0x00000001
    DISABLED = 0x00000002
    NOTPRESENT = 0x00000004
    UNPLUGGED = 0x00000008
    MASK_ALL = 0x0000000F
class AudioDeviceState(Enum):
    Active = 0x1
    Disabled = 0x2
    NotPresent = 0x4
    Unplugged = 0x8
class STGM(Enum):
    STGM_READ = 0x00000000
class AUDCLNT_SHAREMODE(Enum):
    AUDCLNT_SHAREMODE_SHARED = 0x00000001
    AUDCLNT_SHAREMODE_EXCLUSIVE = 0x00000002
class AUDIO_VOLUME_NOTIFICATION_DATA(Structure):
    _fields_ = [
        ('guidEventContext', GUID),
        ('bMuted', BOOL),
        ('fMasterVolume', c_float),
        ('nChannels', UINT),
        ('afChannelVolumes', c_float * 8),
    ]
PAUDIO_VOLUME_NOTIFICATION_DATA = POINTER(AUDIO_VOLUME_NOTIFICATION_DATA)
class IAudioEndpointVolumeCallback(IUnknown):
    _iid_ = GUID('{b1136c83-b6b5-4add-98a5-a2df8eedf6fa}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'OnNotify',
                  (['in'],
                  PAUDIO_VOLUME_NOTIFICATION_DATA,
                  'pNotify')),
    )
class IAudioEndpointVolume(IUnknown):
    _iid_ = GUID('{5CDF2C82-841E-4546-9722-0CF74078229A}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'RegisterControlChangeNotify',
                  (['in'],
                  POINTER(IAudioEndpointVolumeCallback),
                  'pNotify')),
        COMMETHOD([], HRESULT, 'UnregisterControlChangeNotify',
                  (['in'],
                  POINTER(IAudioEndpointVolumeCallback),
                  'pNotify')),
        COMMETHOD([], HRESULT, 'GetChannelCount',
                  (['out'], POINTER(UINT), 'pnChannelCount')),
        COMMETHOD([], HRESULT, 'SetMasterVolumeLevel',
                  (['in'], c_float, 'fLevelDB'),
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'SetMasterVolumeLevelScalar',
                  (['in'], c_float, 'fLevel'),
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'GetMasterVolumeLevel',
                  (['out'], POINTER(c_float), 'pfLevelDB')),
        COMMETHOD([], HRESULT, 'GetMasterVolumeLevelScalar',
                  (['out'], POINTER(c_float), 'pfLevelDB')),
        COMMETHOD([], HRESULT, 'SetChannelVolumeLevel',
                  (['in'], UINT, 'nChannel'),
                  (['in'], c_float, 'fLevelDB'),
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'SetChannelVolumeLevelScalar',
                  (['in'], DWORD, 'nChannel'),
                  (['in'], c_float, 'fLevelDB'),
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'GetChannelVolumeLevel',
                  (['in'], UINT, 'nChannel'),
                  (['out'], POINTER(c_float), 'pfLevelDB')),
        COMMETHOD([], HRESULT, 'GetChannelVolumeLevelScalar',
                  (['in'], DWORD, 'nChannel'),
                  (['out'], POINTER(c_float), 'pfLevelDB')),
        COMMETHOD([], HRESULT, 'SetMute',
                  (['in'], BOOL, 'bMute'),
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'GetMute',
                  (['out'], POINTER(BOOL), 'pbMute')),
        COMMETHOD([], HRESULT, 'GetVolumeStepInfo',
                  (['out'], POINTER(DWORD), 'pnStep'),
                  (['out'], POINTER(DWORD), 'pnStepCount')),
        COMMETHOD([], HRESULT, 'VolumeStepUp',
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'VolumeStepDown',
                  (['in'], POINTER(GUID), 'pguidEventContext')),
        COMMETHOD([], HRESULT, 'QueryHardwareSupport',
                  (['out'], POINTER(DWORD), 'pdwHardwareSupportMask')),
        COMMETHOD([], HRESULT, 'GetVolumeRange',
                  (['out'], POINTER(c_float), 'pfMin'),
                  (['out'], POINTER(c_float), 'pfMax'),
                  (['out'], POINTER(c_float), 'pfIncr')))
class IAudioSessionControl(IUnknown):
    _iid_ = GUID('{F4B1A599-7266-4319-A8CA-E70ACB11E8CD}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'NotImpl1'),
        COMMETHOD([], HRESULT, 'GetDisplayName',
                  (['out'], POINTER(LPWSTR), 'pRetVal')),
        COMMETHOD([], HRESULT, 'NotImpl2'),
        COMMETHOD([], HRESULT, 'NotImpl3'),
        COMMETHOD([], HRESULT, 'NotImpl4'),
        COMMETHOD([], HRESULT, 'NotImpl5'),
        COMMETHOD([], HRESULT, 'NotImpl6'),
        COMMETHOD([], HRESULT, 'NotImpl7'),
        COMMETHOD([], HRESULT, 'NotImpl8'))
class IAudioSessionControl2(IAudioSessionControl):
    _iid_ = GUID('{BFB7FF88-7239-4FC9-8FA2-07C950BE9C6D}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'GetSessionIdentifier',
                  (['out'], POINTER(LPWSTR), 'pRetVal')),
        COMMETHOD([], HRESULT, 'GetSessionInstanceIdentifier',
                  (['out'], POINTER(LPWSTR), 'pRetVal')),
        COMMETHOD([], HRESULT, 'GetProcessId',
                  (['out'], POINTER(DWORD), 'pRetVal')),
        COMMETHOD([], HRESULT, 'IsSystemSoundsSession'),
        COMMETHOD([], HRESULT, 'SetDuckingPreferences',
                  (['in'], BOOL, 'optOut')))
class ISimpleAudioVolume(IUnknown):
    _iid_ = GUID('{87CE5498-68D6-44E5-9215-6DA47EF883D8}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'SetMasterVolume',
                  (['in'], c_float, 'fLevel'),
                  (['in'], POINTER(GUID), 'EventContext')),
        COMMETHOD([], HRESULT, 'GetMasterVolume',
                  (['out'], POINTER(c_float), 'pfLevel')),
        COMMETHOD([], HRESULT, 'SetMute',
                  (['in'], BOOL, 'bMute'),
                  (['in'], POINTER(GUID), 'EventContext')),
        COMMETHOD([], HRESULT, 'GetMute', (['out'], POINTER(BOOL), 'pbMute')))
class IAudioSessionEnumerator(IUnknown):
    _iid_ = GUID('{E2F5BB11-0570-40CA-ACDD-3AA01277DEE8}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'GetCount',
                  (['out'], POINTER(INT), 'SessionCount')),
        COMMETHOD([], HRESULT, 'GetSession',
                  (['in'], INT, 'SessionCount'),
                  (['out'],
                   POINTER(POINTER(IAudioSessionControl)), 'Session')))
class IAudioSessionManager(IUnknown):
    _iid_ = GUID('{BFA971F1-4d5e-40bb-935e-967039bfbee4}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'NotImpl1'),
        COMMETHOD([], HRESULT, 'GetSimpleAudioVolume',
                  (['in'], POINTER(GUID), 'AudioSessionGuid'),
                  (['in'], DWORD, 'CrossProcessSession'),
                  (['out'],
                   POINTER(POINTER(ISimpleAudioVolume)), 'AudioVolume')))
class IAudioSessionManager2(IAudioSessionManager):
    _iid_ = GUID('{77aa99a0-1bd6-484f-8bc7-2c654c9a9b6f}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'GetSessionEnumerator',
                  (['out'],
                  POINTER(POINTER(IAudioSessionEnumerator)), 'SessionList')),
        COMMETHOD([], HRESULT, 'NotImpl1'),
        COMMETHOD([], HRESULT, 'NotImpl2'),
        COMMETHOD([], HRESULT, 'NotImpl1'),
        COMMETHOD([], HRESULT, 'NotImpl2'))
class IAudioClient(IUnknown):
    _iid_ = GUID('{1cb9ad4c-dbfa-4c32-b178-c2f568a703b2}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'Initialize',
                  (['in'], DWORD, 'ShareMode'),
                  (['in'], DWORD, 'StreamFlags'),
                  (['in'], REFERENCE_TIME, 'hnsBufferDuration'),
                  (['in'], REFERENCE_TIME, 'hnsPeriodicity'),
                  (['in'], POINTER(WAVEFORMATEX), 'pFormat'),
                  (['in'], POINTER(GUID), 'AudioSessionGuid')),
        COMMETHOD([], HRESULT, 'GetBufferSize',
                  (['out'], POINTER(UINT32), 'pNumBufferFrames')),
        COMMETHOD([], HRESULT, 'NotImpl1'),
        COMMETHOD([], HRESULT, 'GetCurrentPadding',
                  (['out'], POINTER(UINT32), 'pNumPaddingFrames')),
        COMMETHOD([], HRESULT, 'NotImpl2'),
        COMMETHOD([], HRESULT, 'GetMixFormat',
                  (['out'], POINTER(POINTER(WAVEFORMATEX)), 'ppDeviceFormat')),
        COMMETHOD([], HRESULT, 'NotImpl4'),
        COMMETHOD([], HRESULT, 'Start'),
        COMMETHOD([], HRESULT, 'Stop'),
        COMMETHOD([], HRESULT, 'Reset'),
        COMMETHOD([], HRESULT, 'NotImpl5'),
        COMMETHOD([], HRESULT, 'GetService',
                  (['in'], POINTER(GUID), 'iid'),
                  (['out'], POINTER(POINTER(IUnknown)), 'ppv')))
@python_2_unicode_compatible
class PROPERTYKEY(Structure):
    _fields_ = [
        ('fmtid', GUID),
        ('pid', DWORD),
    ]
    def __str__(self):
        return "%s %s" % (self.fmtid, self.pid)
class IPropertyStore(IUnknown):
    _iid_ = GUID('{886d8eeb-8cf2-4446-8d02-cdba1dbdcf99}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'GetCount',
                  (['out'], POINTER(DWORD), 'cProps')),
        COMMETHOD([], HRESULT, 'GetAt',
                  (['in'], DWORD, 'iProp'),
                  (['out'], POINTER(PROPERTYKEY), 'pkey')),
        COMMETHOD([], HRESULT, 'GetValue',
                  (['in'], POINTER(PROPERTYKEY), 'key'),
                  (['out'], POINTER(PROPVARIANT), 'pv')),
        COMMETHOD([], HRESULT, 'SetValue',
                  (['out'], POINTER(LPWSTR), 'ppstrId')),
        COMMETHOD([], HRESULT, 'Commit'))
class IMMDevice(IUnknown):
    _iid_ = GUID('{D666063F-1587-4E43-81F1-B948E807363F}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'Activate',
                  (['in'], POINTER(GUID), 'iid'),
                  (['in'], DWORD, 'dwClsCtx'),
                  (['in'], POINTER(DWORD), 'pActivationParams'),
                  (['out'],
                   POINTER(POINTER(IUnknown)), 'ppInterface')),
        COMMETHOD([], HRESULT, 'OpenPropertyStore',
                  (['in'], DWORD, 'stgmAccess'),
                  (['out'],
                  POINTER(POINTER(IPropertyStore)), 'ppProperties')),
        COMMETHOD([], HRESULT, 'GetId',
                  (['out'], POINTER(LPWSTR), 'ppstrId')),
        COMMETHOD([], HRESULT, 'GetState',
                  (['out'], POINTER(DWORD), 'pdwState')))
class IMMDeviceCollection(IUnknown):
    _iid_ = GUID('{0BD7A1BE-7A1A-44DB-8397-CC5392387B5E}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'GetCount',
                  (['out'], POINTER(UINT), 'pcDevices')),
        COMMETHOD([], HRESULT, 'Item',
                  (['in'], UINT, 'nDevice'),
                  (['out'], POINTER(POINTER(IMMDevice)), 'ppDevice')))
class IMMDeviceEnumerator(IUnknown):
    _iid_ = GUID('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
    _methods_ = (
        COMMETHOD([], HRESULT, 'EnumAudioEndpoints',
                  (['in'], DWORD, 'dataFlow'),
                  (['in'], DWORD, 'dwStateMask'),
                  (['out'],
                  POINTER(POINTER(IMMDeviceCollection)), 'ppDevices')),
        COMMETHOD([], HRESULT, 'GetDefaultAudioEndpoint',
                  (['in'], DWORD, 'dataFlow'),
                  (['in'], DWORD, 'role'),
                  (['out'], POINTER(POINTER(IMMDevice)), 'ppDevices')),
        COMMETHOD([], HRESULT, 'GetDevice',
                  (['in'], LPCWSTR, 'pwstrId'),
                  (['out'],
                  POINTER(POINTER(IMMDevice)), 'ppDevice')),
        COMMETHOD([], HRESULT, 'NotImpl1'),
        COMMETHOD([], HRESULT, 'NotImpl2'))
@python_2_unicode_compatible
class AudioDevice(object):
    def __init__(self, id, state, properties):
        self.id = id
        self.state = state
        self.properties = properties
    def __str__(self):
        return "AudioDevice: %s" % (self.FriendlyName)
    @property
    def FriendlyName(self):
        DEVPKEY_Device_FriendlyName =            u"{a45c254e-df1c-4efd-8020-67d146a850e0} 14".upper()
        value = self.properties.get(DEVPKEY_Device_FriendlyName)
        return value
@python_2_unicode_compatible
class AudioSession(object):
    def __init__(self, audio_session_control2):
        self._ctl = audio_session_control2
        self._process = None
        self._volume = None
    def __str__(self):
        s = self.DisplayName
        if s:
            return "DisplayName: " + s
        if self.Process is not None:
            return "Process: " + self.Process.name()
        return "Pid: %s" % (self.ProcessId)
    @property
    def Process(self):
        if self._process is None and self.ProcessId != 0:
            try:
                self._process = psutil.Process(self.ProcessId)
            except psutil.NoSuchProcess:
                return None
        return self._process
    @property
    def ProcessId(self):
        return self._ctl.GetProcessId()
    @property
    def Identifier(self):
        s = self._ctl.GetSessionIdentifier()
        return s
    @property
    def InstanceIdentifier(self):
        s = self._ctl.GetSessionInstanceIdentifier()
        return s
    @property
    def State(self):
        s = self._ctl.GetState()
        return s
    @property
    def GroupingParam(self):
        g = self._ctl.GetGroupingParam()
        return g
    @GroupingParam.setter
    def GroupingParam(self, value):
        self._ctl.SetGroupingParam(value, IID_Empty)
    @property
    def DisplayName(self):
        s = self._ctl.GetDisplayName()
        return s
    @DisplayName.setter
    def DisplayName(self, value):
        s = self._ctl.GetDisplayName()
        if s != value:
            self._ctl.SetDisplayName(value, IID_Empty)
    @property
    def IconPath(self):
        s = self._ctl.GetIconPath()
        return s
    @IconPath.setter
    def IconPath(self, value):
        s = self._ctl.GetIconPath()
        if s != value:
            self._ctl.SetIconPath(value, IID_Empty)
    @property
    def SimpleAudioVolume(self):
        if self._volume is None:
            self._volume = self._ctl.QueryInterface(ISimpleAudioVolume)
        return self._volume
class AudioUtilities(object):
    @staticmethod
    def GetSpeakers():
        deviceEnumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER)
        speakers = deviceEnumerator.GetDefaultAudioEndpoint(
                    EDataFlow.eRender.value, ERole.eMultimedia.value)
        return speakers
    @staticmethod
    def GetAudioSessionManager():
        speakers = AudioUtilities.GetSpeakers()
        if speakers is None:
            return None
        o = speakers.Activate(
            IAudioSessionManager2._iid_, comtypes.CLSCTX_ALL, None)
        mgr = o.QueryInterface(IAudioSessionManager2)
        return mgr
    @staticmethod
    def GetAllSessions():
        audio_sessions = []
        mgr = AudioUtilities.GetAudioSessionManager()
        if mgr is None:
            return audio_sessions
        sessionEnumerator = mgr.GetSessionEnumerator()
        count = sessionEnumerator.GetCount()
        for i in range(count):
            ctl = sessionEnumerator.GetSession(i)
            if ctl is None:
                continue
            ctl2 = ctl.QueryInterface(IAudioSessionControl2)
            if ctl2 is not None:
                audio_session = AudioSession(ctl2)
                audio_sessions.append(audio_session)
        return audio_sessions
    @staticmethod
    def GetProcessSession(id):
        for session in AudioUtilities.GetAllSessions():
            if session.ProcessId == id:
                return session
        return None
    @staticmethod
    def CreateDevice(dev):
        if dev is None:
            return None
        id = dev.GetId()
        state = dev.GetState()
        properties = {}
        store = dev.OpenPropertyStore(STGM.STGM_READ.value)
        if store is not None:
            propCount = store.GetCount()
            for j in range(propCount):
                pk = store.GetAt(j)
                value = store.GetValue(pk)
                v = value.GetValue()
                name = str(pk)
                properties[name] = v
        audioState = AudioDeviceState(state)
        return AudioDevice(id, audioState, properties)
    @staticmethod
    def GetAllDevices():
        devices = []
        deviceEnumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER)
        if deviceEnumerator is None:
            return devices
        collection = deviceEnumerator.EnumAudioEndpoints(
            EDataFlow.eAll.value, DEVICE_STATE.MASK_ALL.value)
        if collection is None:
            return devices
        count = collection.GetCount()
        for i in range(count):
            dev = collection.Item(i)
            if dev is not None:
                devices.append(AudioUtilities.CreateDevice(dev))
        return devices
