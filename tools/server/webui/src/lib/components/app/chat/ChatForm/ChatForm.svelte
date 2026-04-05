<script lang="ts">
	import {
		ChatAttachmentsList,
		ChatAttachmentMcpResources,
		ChatFormActions,
		ChatFormFileInputInvisible,
		ChatFormPromptPicker,
		ChatFormResourcePicker,
		ChatFormTextarea
	} from '$lib/components/app';
	import { DialogMcpResources } from '$lib/components/app/dialogs';
	import {
		CLIPBOARD_CONTENT_QUOTE_PREFIX,
		INPUT_CLASSES,
		SETTING_CONFIG_DEFAULT,
		INITIAL_FILE_SIZE,
		PROMPT_CONTENT_SEPARATOR,
		PROMPT_TRIGGER_PREFIX,
		RESOURCE_TRIGGER_PREFIX
	} from '$lib/constants';
	import {
		ContentPartType,
		FileExtensionText,
		KeyboardKey,
		MimeTypeText,
		SpecialFileType
	} from '$lib/enums';
	import { config } from '$lib/stores/settings.svelte';
	import { modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { mcpHasResourceAttachments } from '$lib/stores/mcp-resources.svelte';
	import { conversationsStore, activeMessages } from '$lib/stores/conversations.svelte';
	import type { GetPromptResult, MCPPromptInfo, MCPResourceInfo, PromptMessage } from '$lib/types';
	import { isIMEComposing, parseClipboardContent, uuid } from '$lib/utils';
	import {
		AudioRecorder,
		convertToWav,
		createAudioFile,
		isAudioRecordingSupported
	} from '$lib/utils/browser-only';
	import { onMount } from 'svelte';

	interface Props {
		// Data
		attachments?: DatabaseMessageExtra[];
		uploadedFiles?: ChatUploadedFile[];
		value?: string;

		// UI State
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		placeholder?: string;
		showMcpPromptButton?: boolean;

		// Prompt Prefill
		prefillText?: string;
		onPrefillChange?: (text: string) => void;
		showPrefill?: boolean;

		// Event Handlers
		onAttachmentRemove?: (index: number) => void;
		onFilesAdd?: (files: File[]) => void;
		onStop?: () => void;
		onSubmit?: () => void;
		onSystemPromptClick?: (draft: { message: string; files: ChatUploadedFile[] }) => void;
		onUploadedFileRemove?: (fileId: string) => void;
		onUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		onValueChange?: (value: string) => void;
	}

	let {
		attachments = [],
		class: className = '',
		disabled = false,
		isLoading = false,
		placeholder = 'Type a message...',
		showMcpPromptButton = false,
		prefillText = '',
		onPrefillChange,
		showPrefill = false,
		uploadedFiles = $bindable([]),
		value = $bindable(''),
		onAttachmentRemove,
		onFilesAdd,
		onStop,
		onSubmit,
		onSystemPromptClick,
		onUploadedFileRemove,
		onUploadedFilesChange,
		onValueChange
	}: Props = $props();

	/**
	 *
	 *
	 * STATE
	 *
	 *
	 */

	// Component References
	let audioRecorder: AudioRecorder | undefined;
	let chatFormActionsRef: ChatFormActions | undefined = $state(undefined);
	let fileInputRef: ChatFormFileInputInvisible | undefined = $state(undefined);
	let promptPickerRef: ChatFormPromptPicker | undefined = $state(undefined);
	let resourcePickerRef: ChatFormResourcePicker | undefined = $state(undefined);
	let textareaRef: ChatFormTextarea | undefined = $state(undefined);

	// Audio Recording State
	let isRecording = $state(false);
	let recordingSupported = $state(false);

	// Prompt Picker State
	let isPromptPickerOpen = $state(false);
	let promptSearchQuery = $state('');

	// Inline Resource Picker State (triggered by @)
	let isInlineResourcePickerOpen = $state(false);
	let resourceSearchQuery = $state('');

	// Resource Dialog State
	let isResourceDialogOpen = $state(false);
	let preSelectedResourceUri = $state<string | undefined>(undefined);

	// Prompt Prefill State
	let isPrefillExpanded = $state(false);

	// Prompt Prefill Templates
	const PREFILL_TEMPLATES = [
		{ label: 'JSON', text: 'Respond with only valid JSON, no additional text.\n' },
		{ label: 'XML', text: 'Respond with only valid XML, no additional text.\n' },
		{ label: 'YAML', text: 'Respond with only valid YAML, no additional text.\n' },
		{ label: 'Be concise', text: 'Be concise and direct. Avoid unnecessary details, filler, and preamble.\n' },
		{ label: 'Step by step', text: 'Think through this step by step before providing your final answer.\n' },
		{ label: 'Short answer', text: 'Provide a short, direct answer in 1-2 sentences.\n' },
		{ label: 'Code block', text: 'Wrap your entire response in a single code block.\n' },
		{ label: 'As a table', text: 'Present the information as a markdown table.\n' }
	] as const;

	function handlePrefillInsert(text: string) {
		const newText = prefillText ? prefillText + text : text;
		onPrefillChange?.(newText);
	}

	/**
	 *
	 *
	 * DERIVED STATE
	 *
	 *
	 */

	// Configuration
	let currentConfig = $derived(config());
	let pasteLongTextToFileLength = $derived.by(() => {
		const n = Number(currentConfig.pasteLongTextToFileLen);
		return Number.isNaN(n) ? Number(SETTING_CONFIG_DEFAULT.pasteLongTextToFileLen) : n;
	});

	// Model Selection Logic
	let isRouter = $derived(isRouterMode());
	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);
	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	// Form Validation State
	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());
	let hasLoadingAttachments = $derived(uploadedFiles.some((f) => f.isLoading));
	let hasAttachments = $derived(
		(attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
	);
	let canSubmit = $derived(value.trim().length > 0 || hasAttachments);

	/**
	 *
	 *
	 * LIFECYCLE
	 *
	 *
	 */

	onMount(() => {
		recordingSupported = isAudioRecordingSupported();
		audioRecorder = new AudioRecorder();
	});

	/**
	 *
	 *
	 * PUBLIC API
	 *
	 *
	 */

	export function focus() {
		textareaRef?.focus();
	}

	export function resetTextareaHeight() {
		textareaRef?.resetHeight();
	}

	export function openModelSelector() {
		chatFormActionsRef?.openModelSelector();
	}

	/**
	 * Check if a model is selected, open selector if not
	 * @returns true if model is selected, false otherwise
	 */
	export function checkModelSelected(): boolean {
		if (!hasModelSelected) {
			chatFormActionsRef?.openModelSelector();
			return false;
		}
		return true;
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - File Management
	 *
	 *
	 */

	function handleFileSelect(files: File[]) {
		onFilesAdd?.(files);
	}

	function handleFileUpload() {
		fileInputRef?.click();
	}

	function handleFileRemove(fileId: string) {
		if (fileId.startsWith('attachment-')) {
			const index = parseInt(fileId.replace('attachment-', ''), 10);
			if (!isNaN(index) && index >= 0 && index < attachments.length) {
				onAttachmentRemove?.(index);
			}
		} else {
			onUploadedFileRemove?.(fileId);
		}
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - Input & Keyboard
	 *
	 *
	 */

	function handleInput() {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();
		const hasServers = mcpStore.hasEnabledServers(perChatOverrides);

		if (value.startsWith(PROMPT_TRIGGER_PREFIX) && hasServers) {
			isPromptPickerOpen = true;
			promptSearchQuery = value.slice(1);
			isInlineResourcePickerOpen = false;
			resourceSearchQuery = '';
		} else if (
			value.startsWith(RESOURCE_TRIGGER_PREFIX) &&
			hasServers &&
			mcpStore.hasResourcesCapability(perChatOverrides)
		) {
			isInlineResourcePickerOpen = true;
			resourceSearchQuery = value.slice(1);
			isPromptPickerOpen = false;
			promptSearchQuery = '';
		} else {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
			isInlineResourcePickerOpen = false;
			resourceSearchQuery = '';
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (isPromptPickerOpen && promptPickerRef?.handleKeydown(event)) {
			return;
		}

		if (isInlineResourcePickerOpen && resourcePickerRef?.handleKeydown(event)) {
			return;
		}

		if (event.key === KeyboardKey.ESCAPE && isPromptPickerOpen) {
			isPromptPickerOpen = false;
			promptSearchQuery = '';
			return;
		}

		if (event.key === KeyboardKey.ESCAPE && isInlineResourcePickerOpen) {
			isInlineResourcePickerOpen = false;
			resourceSearchQuery = '';
			return;
		}

		if (event.key === KeyboardKey.ENTER && !event.shiftKey && !isIMEComposing(event)) {
			event.preventDefault();

			if (!canSubmit || disabled || isLoading || hasLoadingAttachments) return;

			onSubmit?.();
		}
	}

	function handlePaste(event: ClipboardEvent) {
		if (!event.clipboardData) return;

		const files = Array.from(event.clipboardData.items)
			.filter((item) => item.kind === 'file')
			.map((item) => item.getAsFile())
			.filter((file): file is File => file !== null);

		if (files.length > 0) {
			event.preventDefault();
			onFilesAdd?.(files);
			return;
		}

		const text = event.clipboardData.getData(MimeTypeText.PLAIN);

		if (text.startsWith(CLIPBOARD_CONTENT_QUOTE_PREFIX)) {
			const parsed = parseClipboardContent(text);

			if (parsed.textAttachments.length > 0 || parsed.mcpPromptAttachments.length > 0) {
				event.preventDefault();
				value = parsed.message;
				onValueChange?.(parsed.message);

				// Handle text attachments as files
				if (parsed.textAttachments.length > 0) {
					const attachmentFiles = parsed.textAttachments.map(
						(att) =>
							new File([att.content], att.name, {
								type: MimeTypeText.PLAIN
							})
					);
					onFilesAdd?.(attachmentFiles);
				}

				// Handle MCP prompt attachments as ChatUploadedFile with mcpPrompt data
				if (parsed.mcpPromptAttachments.length > 0) {
					const mcpPromptFiles: ChatUploadedFile[] = parsed.mcpPromptAttachments.map((att) => ({
						id: uuid(),
						name: att.name,
						size: att.content.length,
						type: SpecialFileType.MCP_PROMPT,
						file: new File([att.content], `${att.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						}),
						isLoading: false,
						textContent: att.content,
						mcpPrompt: {
							serverName: att.serverName,
							promptName: att.promptName,
							arguments: att.arguments
						}
					}));

					uploadedFiles = [...uploadedFiles, ...mcpPromptFiles];
					onUploadedFilesChange?.(uploadedFiles);
				}

				setTimeout(() => {
					textareaRef?.focus();
				}, 10);

				return;
			}
		}

		if (
			text.length > 0 &&
			pasteLongTextToFileLength > 0 &&
			text.length > pasteLongTextToFileLength
		) {
			event.preventDefault();

			const textFile = new File([text], 'Pasted', {
				type: MimeTypeText.PLAIN
			});

			onFilesAdd?.([textFile]);
		}
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - Prompt Picker
	 *
	 *
	 */

	function handlePromptLoadStart(
		placeholderId: string,
		promptInfo: MCPPromptInfo,
		args?: Record<string, string>
	) {
		// Only clear the value if the prompt was triggered by typing '/'
		if (value.startsWith(PROMPT_TRIGGER_PREFIX)) {
			value = '';
			onValueChange?.('');
		}
		isPromptPickerOpen = false;
		promptSearchQuery = '';

		const promptName = promptInfo.title || promptInfo.name;
		const placeholder: ChatUploadedFile = {
			id: placeholderId,
			name: promptName,
			size: INITIAL_FILE_SIZE,
			type: SpecialFileType.MCP_PROMPT,
			file: new File([], 'loading'),
			isLoading: true,
			mcpPrompt: {
				serverName: promptInfo.serverName,
				promptName: promptInfo.name,
				arguments: args ? { ...args } : undefined
			}
		};

		uploadedFiles = [...uploadedFiles, placeholder];
		onUploadedFilesChange?.(uploadedFiles);
		textareaRef?.focus();
	}

	function handlePromptLoadComplete(placeholderId: string, result: GetPromptResult) {
		const promptText = result.messages
			?.map((msg: PromptMessage) => {
				if (typeof msg.content === 'string') {
					return msg.content;
				}

				if (msg.content.type === ContentPartType.TEXT) {
					return msg.content.text;
				}

				return '';
			})
			.filter(Boolean)
			.join(PROMPT_CONTENT_SEPARATOR);

		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId
				? {
						...f,
						isLoading: false,
						textContent: promptText,
						size: promptText.length,
						file: new File([promptText], `${f.name}${FileExtensionText.TXT}`, {
							type: MimeTypeText.PLAIN
						})
					}
				: f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	function handlePromptLoadError(placeholderId: string, error: string) {
		uploadedFiles = uploadedFiles.map((f) =>
			f.id === placeholderId ? { ...f, isLoading: false, loadError: error } : f
		);
		onUploadedFilesChange?.(uploadedFiles);
	}

	function handlePromptPickerClose() {
		isPromptPickerOpen = false;
		promptSearchQuery = '';
		textareaRef?.focus();
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - Inline Resource Picker
	 *
	 *
	 */

	function handleInlineResourcePickerClose() {
		isInlineResourcePickerOpen = false;
		resourceSearchQuery = '';
		textareaRef?.focus();
	}

	function handleInlineResourceSelect() {
		// Clear the @query from input after resource is attached
		if (value.startsWith(RESOURCE_TRIGGER_PREFIX)) {
			value = '';
			onValueChange?.('');
		}

		isInlineResourcePickerOpen = false;
		resourceSearchQuery = '';
		textareaRef?.focus();
	}

	function handleBrowseResources() {
		isInlineResourcePickerOpen = false;
		resourceSearchQuery = '';

		if (value.startsWith(RESOURCE_TRIGGER_PREFIX)) {
			value = '';
			onValueChange?.('');
		}

		isResourceDialogOpen = true;
	}

	/**
	 *
	 *
	 * EVENT HANDLERS - Audio Recording
	 *
	 *
	 */

	async function handleMicClick() {
		if (!audioRecorder || !recordingSupported) {
			console.warn('Audio recording not supported');
			return;
		}

		if (isRecording) {
			try {
				const audioBlob = await audioRecorder.stopRecording();
				const wavBlob = await convertToWav(audioBlob);
				const audioFile = createAudioFile(wavBlob);

				onFilesAdd?.([audioFile]);
				isRecording = false;
			} catch (error) {
				console.error('Failed to stop recording:', error);
				isRecording = false;
			}
		} else {
			try {
				await audioRecorder.startRecording();
				isRecording = true;
			} catch (error) {
				console.error('Failed to start recording:', error);
			}
		}
	}
</script>

<ChatFormFileInputInvisible bind:this={fileInputRef} onFileSelect={handleFileSelect} />

<form
	class="relative {className}"
	onsubmit={(e) => {
		e.preventDefault();
		if (!canSubmit || disabled || isLoading || hasLoadingAttachments) return;
		onSubmit?.();
	}}
>
	<ChatFormPromptPicker
		bind:this={promptPickerRef}
		isOpen={isPromptPickerOpen}
		searchQuery={promptSearchQuery}
		onClose={handlePromptPickerClose}
		onPromptLoadStart={handlePromptLoadStart}
		onPromptLoadComplete={handlePromptLoadComplete}
		onPromptLoadError={handlePromptLoadError}
	/>

	<ChatFormResourcePicker
		bind:this={resourcePickerRef}
		isOpen={isInlineResourcePickerOpen}
		searchQuery={resourceSearchQuery}
		onClose={handleInlineResourcePickerClose}
		onResourceSelect={handleInlineResourceSelect}
		onBrowse={handleBrowseResources}
	/>

	<div
		class="{INPUT_CLASSES} overflow-hidden rounded-3xl backdrop-blur-md {disabled
			? 'cursor-not-allowed opacity-60'
			: ''}"
		data-slot="input-area"
	>
		<ChatAttachmentsList
			{attachments}
			bind:uploadedFiles
			onFileRemove={handleFileRemove}
			limitToSingleRow
			class="py-5"
			style="scroll-padding: 1rem;"
			activeModelId={activeModelId ?? undefined}
		/>

		<div
			class="flex-column relative min-h-[48px] items-center rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:!py-3"
			onpaste={handlePaste}
		>
			<ChatFormTextarea
				class="px-5 py-1.5 md:pt-0"
				bind:this={textareaRef}
				bind:value
				onKeydown={handleKeydown}
				onInput={() => {
					handleInput();
					onValueChange?.(value);
				}}
				{disabled}
				{placeholder}
			/>

			{#if mcpHasResourceAttachments()}
				<ChatAttachmentMcpResources
					class="mb-3"
					onResourceClick={(uri) => {
						preSelectedResourceUri = uri;
						isResourceDialogOpen = true;
					}}
				/>
			{/if}

			<ChatFormActions
				class="px-3"
				bind:this={chatFormActionsRef}
				canSend={canSubmit}
				hasText={value.trim().length > 0}
				{disabled}
				{isLoading}
				{isRecording}
				{uploadedFiles}
				onFileUpload={handleFileUpload}
				onMicClick={handleMicClick}
				{onStop}
				onSystemPromptClick={() => onSystemPromptClick?.({ message: value, files: uploadedFiles })}
				onMcpPromptClick={showMcpPromptButton ? () => (isPromptPickerOpen = true) : undefined}
				onMcpResourcesClick={() => (isResourceDialogOpen = true)}
			/>
		</div>
	</div>
</form>

{#if showPrefill}
	<div class="mx-auto mt-2 max-w-[48rem]">
		<button
			type="button"
			class="flex w-full items-center gap-2 px-4 py-2 text-sm text-muted-foreground hover:text-foreground"
			onclick={() => (isPrefillExpanded = !isPrefillExpanded)}
		>
			<svg
				class="h-4 w-4 transition-transform {isPrefillExpanded ? 'rotate-90' : ''}"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
			>
				<polyline points="9 18 15 12 9 6" />
			</svg>
			<span>Prompt Prefill</span>
		</button>

		{#if isPrefillExpanded}
			<!-- Prefill Template Buttons -->
			<div class="mb-2 flex flex-wrap gap-1.5 px-4">
				{#each PREFILL_TEMPLATES as tpl (tpl.label)}
					<button
						type="button"
						class="prefill-template-pill"
						onclick={() => handlePrefillInsert(tpl.text)}
					>
						{tpl.label}
					</button>
				{/each}
			</div>
			<textarea
				class="w-full rounded-lg border border-border bg-background px-4 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
				rows="3"
				placeholder="Text to prefill the assistant's response..."
				value={prefillText}
				oninput={(e) => onPrefillChange?.((e.target as HTMLTextAreaElement).value)}
			></textarea>
		{/if}
	</div>
{/if}

<DialogMcpResources
	bind:open={isResourceDialogOpen}
	preSelectedUri={preSelectedResourceUri}
	onAttach={(resource: MCPResourceInfo) => {
		mcpStore.attachResource(resource.uri);
	}}
	onOpenChange={(newOpen: boolean) => {
		if (!newOpen) {
			preSelectedResourceUri = undefined;
		}
	}}
/>
