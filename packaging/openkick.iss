; Inno Setup script for the Windows installer.
;   ISCC.exe /DDistDir=<folder containing OpenKick\> /DAppVersion=0.1.0 packaging\openkick.iss
#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#ifndef DistDir
  #define DistDir "..\dist"
#endif

[Setup]
AppId={{5C2E8A4B-7F1D-4E3A-9B6C-0A8D2F4E6B11}
AppName=OpenKick
AppVersion={#AppVersion}
AppVerName=OpenKick {#AppVersion}
AppPublisher=Gabriel Alexander Laurente (SlightlyDazed)
AppCopyright=Copyright (C) 2026 Gabriel Alexander Laurente (SlightlyDazed)
AppComments=Análisis biomecánico de video con detección automática de la pose
DefaultDirName={autopf}\OpenKick
DefaultGroupName=OpenKick
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=OpenKick-{#AppVersion}-Setup
SetupIconFile=..\openkick\assets\openkick.ico
UninstallDisplayIcon={app}\OpenKick.exe
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes

[Languages]
; the installer's own texts (the program's texts are in openkick/locale/)
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "german"; MessagesFile: "compiler:Languages\German.isl"
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "portuguese"; MessagesFile: "compiler:Languages\Portuguese.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
; from jrsoftware.org/files/istrans (official since Inno Setup 6.5, not shipped with every install)
Name: "chinesesimplified"; MessagesFile: "languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#DistDir}\OpenKick\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\OpenKick"; Filename: "{app}\OpenKick.exe"
Name: "{autodesktop}\OpenKick"; Filename: "{app}\OpenKick.exe"; Tasks: desktopicon

[Registry]
; the program starts in the language chosen here (QSettings), unless one was already chosen in it
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "es"; Languages: spanish; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "en"; Languages: english; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "de"; Languages: german; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "fr"; Languages: french; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "it"; Languages: italian; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "pt_BR"; Languages: brazilianportuguese; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "pt_BR"; Languages: portuguese; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "ru"; Languages: russian; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "ja"; Languages: japanese; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "ko"; Languages: korean; Flags: createvalueifdoesntexist
Root: HKCU; Subkey: "Software\OpenKick\OpenKick"; ValueType: string; ValueName: "language"; ValueData: "zh_CN"; Languages: chinesesimplified; Flags: createvalueifdoesntexist
Root: HKA; Subkey: "Software\Classes\.okick"; ValueType: string; ValueName: ""; ValueData: "OpenKick.Project"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\OpenKick.Project"; ValueType: string; ValueName: ""; ValueData: "OpenKick project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\OpenKick.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\OpenKick.exe,0"
Root: HKA; Subkey: "Software\Classes\OpenKick.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\OpenKick.exe"" ""%1"""

[Run]
Filename: "{app}\OpenKick.exe"; Description: "{cm:LaunchProgram,OpenKick}"; Flags: nowait postinstall skipifsilent
