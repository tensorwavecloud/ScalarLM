gotemplate
{{- define "cray.fullname" -}}
{{- printf "%s" .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cray.labels" -}}
app.kubernetes.io/name: {{ include "cray.fullname" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
